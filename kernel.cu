#include "kernelCall.h"
#include <cmath>

#define EPSILON (10e-6)
#define ABS(X) ((X) < 0 ? (-X) : (X))


__global__ void Kernel1
(float* _jojo, B* _aliance,
	int* _alianceID, float* _dist,
	int _jojoSize, int _alianceSize, int numPart, float rangePart)
{
	int idx = 0;
	int idy = 0;
	int idz = 0;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= _jojoSize)
		return;

	int s_alianceID = -1;
	float min = 1024 * 1024 * 1024.0;
	
	float j_x, j_y, j_z;
	j_x = _jojo[gid]; j_y = _jojo[gid + _jojoSize]; j_z = _jojo[gid + _jojoSize*2];

	idx = (int)(j_x / rangePart);
	idy = (int)(j_y / rangePart);
	idz = (int)(j_z / rangePart);

	float dist = 0.0;
	for (int z = idz - 1; z <= idz + 1; z++)
	{
		for (int y = idy - 1; y <= idy + 1; y++)
		{
			for (int x = idx - 1; x <= idx + 1; x++)
			{
				if (x < 0 || x >= numPart || y < 0 || y >= numPart || z < 0 || z >= numPart)
				{
					continue;
				}
				for (int i = 0; i < _aliance[z * numPart * numPart + y * numPart + x].n; i++)
				{
					dist = 0.0;
					dist += (j_x - _aliance[z * numPart * numPart + y * numPart + x].all[i].x) * (j_x - _aliance[z * numPart * numPart + y * numPart + x].all[i].x);
					dist += (j_y - _aliance[z * numPart * numPart + y * numPart + x].all[i].y) * (j_y - _aliance[z * numPart * numPart + y * numPart + x].all[i].y);
					dist += (j_z - _aliance[z * numPart * numPart + y * numPart + x].all[i].z) * (j_z - _aliance[z * numPart * numPart + y * numPart + x].all[i].z);

					if (dist < min)
					{
						min = dist;
						s_alianceID = _aliance[z * numPart * numPart + y * numPart + x].all[i].id;
					}
				}
			}
		}
	}
	
	_alianceID[gid] = s_alianceID;
	_dist[gid] = min;
}


bool kernelCall
(float* jojo, B* aliance,
	int* alianceID, float* dist,
	int jojoSize, int alianceSize, int numPart, float rangePart, int _layout)
{

	switch (_layout)
	{
	case ThreadLayout::KERNEL_1:
	{
		dim3 gridDim(ceil(jojoSize / 1024.0));
		dim3 blockDim(1024);
		Kernel1 << <gridDim, blockDim >> > (
			jojo, aliance,
			alianceID, dist,
			jojoSize, alianceSize, numPart, rangePart);
		cudaDeviceSynchronize();
		break;
	}
	default:
		printf("Not supported layout\n");
		return false;
	}
	return true;
}