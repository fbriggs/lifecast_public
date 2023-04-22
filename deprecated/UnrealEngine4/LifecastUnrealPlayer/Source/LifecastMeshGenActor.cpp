/*
The MIT License

Copyright © 2021 Lifecast Incorporated

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include "LifecastMeshGenActor.h"
#include "Math/UnrealMathUtility.h"

// Sets default values
ALifecastMeshGenActor::ALifecastMeshGenActor()
{
	mesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("GeneratedMesh"));
	RootComponent = mesh;

	material = CreateDefaultSubobject<UMaterial>(TEXT("material"));
}

// Called when the game starts or when spawned
void ALifecastMeshGenActor::BeginPlay()
{
	Super::BeginPlay();

	constexpr int GRID_SIZE = 2048;
	constexpr float FTHETA_SCALE = 1.2;

	TArray<FVector> vertices;
	TArray<int32> triangles;
	TArray<FVector2D> uv0;
	
	TArray<FVector> normals; // unused
	TArray<FProcMeshTangent> tangents;  // unused
	TArray<FLinearColor> vertexColors;  // unused

	for (int j = 0; j <= GRID_SIZE; ++j) {
		for (int i = 0; i <= GRID_SIZE; ++i) {
			float u = (float)i / (float)GRID_SIZE;
			float v = (float)j / (float)GRID_SIZE;
			float a = 2.0f * (u - 0.5f);
			float b = 2.0f * (v - 0.5f);
			float theta = atan2(b, a);

			float r = sqrt(a * a + b * b) / FTHETA_SCALE;
			float phi = r * PI/ 2.0f;

			float x = cos(phi);
			float y = cos(theta) * sin(phi);
			float z = -sin(theta) * sin(phi); 

			vertices.Add(FVector(x, y, z));
			uv0.Add(FVector2D(u, v));
		}
	}

	for (int j = 0; j < GRID_SIZE; ++j) {
		for (int i = 0; i < GRID_SIZE; ++i) {
			int di = i - GRID_SIZE / 2;
			int dj = j - GRID_SIZE / 2;
			if (di * di + dj * dj > GRID_SIZE * GRID_SIZE / 4) continue;

			int a = i + (GRID_SIZE + 1) * j;
			int b = a + 1;
			int c = a + (GRID_SIZE + 1);
			int d = c + 1;

			triangles.Add(a);
			triangles.Add(c);
			triangles.Add(b);

			triangles.Add(c);
			triangles.Add(d);
			triangles.Add(b);
		}
	}

	mesh->CreateMeshSection_LinearColor(0, vertices, triangles, normals, uv0, vertexColors, tangents, false);
	mesh->SetBoundsScale(10000.0); // This is a hack to prevent frustum culling from hiding the whole mesh (because without warping in the shader, its only 1cm and can easily be outside the view frustum)
	mesh->SetRenderCustomDepth(true);
	mesh->SetMaterial(0, material);
}

