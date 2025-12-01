What the run names mean:

on Intel hardware:

target is "alchemist" or "battlemage"

"wmma" == simple wmma loop, no fancy tiliny.

"tiled" == tiled layout, not using shared memory
"fake1" == tiled variant, but loop structure alterd to fit shared memory layout
"fake2" == tiled variant, but writing data to shared memory
"shmem" == tiled strategy, with shared memory being used

"blocked" == blocked strategy, using shared memory

on NVIDIA hardware:

target is "rtx-xxyy"

"blocked" == blocked strategy, using shared memory
"tiled" == tiled layout, not using shared memory
"wmma" == simple wmma loop, no fancy tiling.


Collumns:

"pb" == parallel blocks. Some layouts launch a number of workgroups relative to the number of SMs on the hardware. This factor scales that number of workgroups constantly, i.e. N_workgroups = N_sm * pb
        In the WMMA implementation, this factor is used to pack multiple tiles in one workgroup, to improve utilization.
"ks" == k_shared_tiles. If shared memory is used, load this many tiles in the K dimension in each iteration. The M and N dimentions are taken from the block layout.
"sk" == skew. Individual lines/columns in shared memory are padded by this many f16 elements to reduce cache collisions.
"x", "y" == Number of tiles in the N and M dimension that each *workgroup* will compute.
"wx", "wy" == Number of tiles in the N and M dimension that each *subgroup* will compute.
"lb" == Layout of the B matrix. The A and C matrix only support row-major for most of the implementations, so that was chosen for all benchmarks until now.
"time", "Tflops" == self explanatory. The Tflops are currently calculated assuming N * M * K * 2 operations, and N = M = K = 4096. Additional columns might be introduced once that changes.
