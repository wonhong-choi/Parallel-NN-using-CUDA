#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct marin
{
public:
	float x, y, z;
	int id;
	marin(float x = -1, float y = -1, float z = -1, int id = -1)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->id = id;
	}
};

struct B
{
public:
	int n = 0;
	marin all[10];

	void push(float x, float y, float z, int id)
	{
		if (n >= 10)
		{
			return;
		}
		all[n] = marin(x, y, z, id);
		n++;
	}
};

enum ThreadLayout
{
	KERNEL_1 = 0x00, // GLOBAL
	KERNEL_2 = 0x01 // SHARED
};

/**
Interface function for kernel call
*/

bool kernelCall
(float* jojo, B* aliance,
	int* alianceID, float* dist,
	int jojoSize, int alianceSize, int numPart, float rangePart, int _layout);