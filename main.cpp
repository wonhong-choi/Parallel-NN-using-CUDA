#include "../Common/BigWar.h"
#include "../Common/DS_timer.h"
#include "Grader.h"
#include "kernelCall.h"
#include <vector>
#include <algorithm>
#include <omp.h>

#include <queue>	// for priority queue

using namespace std;

#define _CRT_SECURE_NO_WARNINGS

#define NUM_THREADS 8



bool kernel13(float* jojo, B* aliance,
	int* alianceID, float* distID,
	int jojoSize, int alianceSize, int numPart, float rangePart)
{
	return kernelCall(jojo, aliance,
		alianceID, distID,
		jojoSize, alianceSize, numPart, rangePart, ThreadLayout::KERNEL_1);
}

struct cmp {
public:
	bool operator()(const Pair& lhs, const Pair& rhs) {
		if (std::abs((POS_TYPE)(lhs.dist - rhs.dist)) < EPSILON) { // equidistant  
			if (lhs.A == rhs.A)
				return lhs.B < rhs.B;
			return lhs.A < rhs.A;
		}
		return lhs.dist < rhs.dist;
	}
};


int main(int argc, char** argv) {

	if (argc < 4) {
		printf("Usage: exe inputA inputB outputFile GT_file(optional for grading) TeamID(optional for grading)\n");
		exit(1);
	}

	DS_timer timer(10);
	timer.initTimers();
	timer.setTimerName(0, "ALL");


	Pair result[100];

	timer.onTimer(0);

	// **************************************//
	// Write your code here
	// CAUTION: DO NOT MODITY OTHER PART OF THE main() FUNCTION

	FILE* fp = NULL;
	UINT numArmies = 0;
	
	float* jojo;
	int jojoSize;
	
	int alianceSize;
	UINT numPart;
	float rangePart;

	// read A
	fopen_s(&fp, argv[1], "rb");
	if (fp == NULL) {
		printf("Fail to read the file - %s\n", argv[1]);
		exit(2);
	}

	fread_s(&numArmies, sizeof(UINT), sizeof(UINT), 1, fp);
	jojoSize = numArmies;
	
	jojo = (float*)malloc(sizeof(float) * numArmies * 3);
	for (int pos = 0; pos < numArmies; pos++) {
		float buf[3];
		if (fread_s(buf, sizeof(float) * 3, sizeof(float), 3, fp) == 0) {
			break;
		}
		for (int i = 0; i < 3; i++) {
			jojo[jojoSize * i + pos] = buf[i];
		}
	}
	
	// read B
	fopen_s(&fp, argv[2], "rb");
	if (fp == NULL) {
		printf("Fail to read the file - %s\n", argv[2]);
		exit(2);
	}

	fread_s(&numArmies, sizeof(UINT), sizeof(UINT), 1, fp);
	alianceSize = numArmies;
	numPart = (UINT)cbrt(numArmies);
	rangePart = ((RANGE_MAX + 1) / (float)numPart);


	B* cbNum = new B[numPart * numPart * numPart];

	int index = 0;
	for (int pos = 0; pos < numArmies; pos++) {
		float buf[3];
		if (fread_s(buf, sizeof(float) * 3, sizeof(float), 3, fp) == 0) {
			break;
		}

		index = (int)(buf[2] / rangePart) * numPart * numPart + (int)(buf[1] / rangePart) * numPart + (int)(buf[0] / rangePart);
		cbNum[index].push(buf[0], buf[1], buf[2], pos);
	}
	fclose(fp);


	float* d_jojo;
	B* d_aliance;

	int* alianceID;
	float* distID;
	int sizeID = sizeof(int) * jojoSize;
	cudaMalloc(&alianceID, sizeID);
	cudaMalloc(&distID, sizeID);

	int jojoMemSize = sizeof(float) * jojoSize * 3;
	cudaMalloc(&d_jojo, jojoMemSize);
	cudaMemcpy(d_jojo, jojo, jojoMemSize, cudaMemcpyHostToDevice);

	cudaMalloc(&d_aliance, sizeof(B) * numPart * numPart * numPart);
	cudaMemcpy(d_aliance, cbNum, sizeof(B) * numPart * numPart * numPart, cudaMemcpyHostToDevice);

	kernel13(d_jojo, d_aliance,
		alianceID, distID,
		jojoSize, alianceSize, numPart, rangePart);

	int* aliance_arr = new int[jojoSize];
	float* dist_arr = new float[jojoSize];

	cudaMemcpy(aliance_arr, alianceID, sizeof(int) * jojoSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(dist_arr, distID, sizeof(float) * jojoSize, cudaMemcpyDeviceToHost);


	// heap
	priority_queue<Pair, vector<Pair>, cmp> pq;

	for (int i = 0; i < 100; ++i) {
		pq.push(Pair(i, aliance_arr[i], dist_arr[i]));
	}
	for (int i = 100; i < jojoSize; ++i) {
		if (pq.top().dist > dist_arr[i]) {
			pq.pop();
			pq.push(Pair(i, aliance_arr[i], dist_arr[i]));
		}
	}

	int i = 99;
	while (!pq.empty()) {
		//printf("%10d %10d %10f\n", pq.top().A, pq.top().B, pq.top().dist);
		result[i--] = pq.top();
		pq.pop();
	}


#pragma endregion
	delete[] aliance_arr;
	delete[] dist_arr;
	delete[] cbNum;

	free(jojo);

	cudaFree(d_jojo);
	cudaFree(d_aliance);
	cudaFree(alianceID);
	cudaFree(distID);

	// print result
	fp = NULL;
	fopen_s(&fp, argv[3], "w");
	if (fp == NULL) {
		printf("Fail to open %s\n", argv[3]);
		exit(3);
	}

	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < 100; ++i) {
		result[i].dist = sqrt(result[i].dist);
	}
	
	for (int i = 0; i < 100; ++i) {
		fprintf(fp, "%d %d %.2f\n", result[i].A, result[i].B, result[i].dist);
	}


	//***************************************//
	timer.offTimer(0);
	timer.printTimer(0);

	// Result validation
	if (argc < 5)
		return 0;

	// Grading mode
	if (argc < 6) {
		printf("Not enough argument for grading\n");
		exit(2);
	}

	Grader grader(argv[4]);
	grader.grading(result);

	fp = NULL;
	fopen_s(&fp, argv[5], "a");
	if (fp == NULL) {
		printf("Fail to open %s\n", argv[5]);
		exit(3);
	}
	fprintf(fp, "%f\t%d\n", timer.getTimer_ms(0), grader.getNumCorrect());
	fclose(fp);
	return 0;
}