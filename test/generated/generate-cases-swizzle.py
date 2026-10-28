from itertools import product

def translate(layout):
    if layout == "row":
        return "AddrMode::RowMajor"
    elif layout == "col":
        return "AddrMode::ColMajor"

def generate_test_layout(name, n, l1, l2, l3):
    with open("%s_labc_%s_%s_%s.art" % (name, l1, l2, l3), "w+") as f:
        f.write("static N = " + str(n) + ";\n")
        f.write("static M = " + str(n) + ";\n")
        f.write("static K = " + str(n) + ";\n")

        f.write("static alayout = " + translate(l1) + ";\n")
        f.write("static blayout = " + translate(l2) + ";\n")
        f.write("static clayout = " + translate(l3) + ";\n")

def generate_test_all(name, x, y, xw, yw, pb, l1, l2, l3, kshare, skew, sas, sad, sbs, sbd):
    n = 4096

    with open("%s_bxy_%s_%s_wxy_%s_%s_pb_%s_labc_%s_%s_%s_ks_%s_sk_%s_sa_%s_%s_sb_%s_%s.art" % (name, x, y, xw, yw, pb, l1, l2, l3, kshare, skew, sas, sad, sbs, sbd), "w+") as f:
        f.write("static N = " + str(n) + ";\n")
        f.write("static M = " + str(n) + ";\n")
        f.write("static K = " + str(n) + ";\n")

        f.write("static alayout = " + translate(l1) + ";\n")
        f.write("static blayout = " + translate(l2) + ";\n")
        f.write("static clayout = " + translate(l3) + ";\n")

        f.write("static swizzle_a_static = " + str(sas) + ";\n")
        f.write("static swizzle_a_dynamic = " + str(sad) + ";\n")
        f.write("static swizzle_b_static = " + str(sbs) + ";\n")
        f.write("static swizzle_b_dynamic = " + str(sbd) + ";\n")

        f.write("static block_x_tiles = " + str(x) + ";\n")
        f.write("static block_y_tiles = " + str(y) + ";\n")

        f.write("static warp_x_tiles = "  + str(xw) + ";\n")
        f.write("static warp_y_tiles = "  + str(yw) + ";\n")

        f.write("static k_shared_tiles = " + str(kshare) + ";\n")
        f.write("static skew_half = " + str(skew) + ";\n")

        f.write("static pb_factor = " + str(pb) + ";\n")


layouts = ["row", "col"]
block_size = [2, 4, 8]
warp_size = [2, 4, 8]
k_shared_tiles = [1, 2, 4]
skew_half = [0, 16]
pb_factor = [2, 4, 8]

#block_size = [8]
#warp_size = [4]
#k_shared_tiles = [2, 4]
#skew_half = [16]
#pb_factor = [4]

"""
Used to test blas.

for l1, l2 in product(layouts, layouts):
    generate_test_layout("test", 256, l1, l2, "col")
    generate_test_layout("bench", 4096, l1, l2, "col")
"""

"""
Used to test the tiled_shmem variant.
"""

a_static = [0, 1, 2, 3]
a_dynamic = [0, 1, 2, 3]
b_static = [0, 1, 2, 3]
b_dynamic = [0, 1, 2, 3]

#for l2, x, y, xw, yw, kshare, skew, pb, sas, sad, sbs, sbd in product(layouts, block_size, block_size, warp_size, warp_size, k_shared_tiles, skew_half, pb_factor):
for sas, sad, sbs, sbd in product(a_static, a_dynamic, b_static, b_dynamic):
    l2 = "col"
    x = 4
    y = 4
    xw = 1
    yw = 2
    kshare = 2
    skew = 0
    pb = 4

    l1 = "row"
    l3 = "row"

    if x % xw != 0:
        continue
    if y % yw != 0:
        continue

    if (x // xw) * (y // yw) <= 1: #We need at least two warps for copying.
        continue

    #generate_test_all("bench", x, y, xw, yw, pb, l1, l2, l3, kshare, skew)
    generate_test_all("test", x, y, xw, yw, pb, l1, l2, l3, kshare, skew, sas, sad, sbs, sbd)

"""
Used for testing the wmma variant.

for l2, pb in product(layouts, pb_factor):
    x = 4
    y = 4
    xw = 2
    yw = 2
    kshare = 2
    skew = 16
    l1 = "row"
    l3 = "row"

    #if x % xw != 0:
    #    continue
    #if y % yw != 0:
    #    continue

    #if (x // xw) * (y // yw) <= 1: #We need at least two warps for copying.
    #    continue

    generate_test_all("bench", x, y, xw, yw, pb, l1, l2, l3, kshare, skew)
    generate_test_all("test", x, y, xw, yw, pb, l1, l2, l3, kshare, skew)
"""

#sizes = [
#        64,
#        128,
#        256,
#        512,
#        512 + 256,
#        1024,
#        1024 + 256,
#        1024 + 512,
#        1024 + 512 + 256,
#        2048,
#        2048 + 512,
#        2048 + 1024,
#        2048 + 1024 + 512,
#        4096,
#        4096 + 512,
#        4096 + 1024,
#        4096 + 1024 + 512,
#        4096 + 2048,
#        4096 + 2048 + 512,
#        4096 + 2048 + 1024,
#        4096 + 2048 + 1024 + 512,
#        8192
#         ]

#def stringfix(n):
#    k = str(i)
#    if len(k) == 1:
#        return "0" + k
#    else:
#        return k

#for i in range(0, len(sizes)):
#    generate_test(stringfix(i), str(sizes[i]))
