from itertools import product

sizes = ["16", "32", "64"]
layouts = ["row", "col"]

def translate(layout):
    if layout == "row":
        return "AddrMode::RowMayor"
    elif layout == "col":
        return "AddrMode::ColMayor"

def generate_test(n, m, k, al, bl, cl):
    with open("case_%s_%s_%s_%s_%s_%s.art" % (n, m, k, al, bl, cl), "w+") as f:
        f.write("static N = " + n + ";\n")
        f.write("static M = " + m + ";\n")
        f.write("static K = " + k + ";\n")
        f.write("static alayout = " + translate(al) + ";\n")
        f.write("static blayout = " + translate(bl) + ";\n")
        f.write("static clayout = " + translate(cl) + ";\n")

for args in product(sizes, sizes, sizes, layouts, layouts, layouts):
    generate_test(*args)
