from itertools import product

prefix = ["", "@"]
namespace = ["Generic", "Global"]
acc_version = ["tiled"]
warmup_version = ["tiled", "simple", "wmma"]

def translate_prefix(a):
    if a == "":
        return "fl"
    else:
        return "pe"

for ns_acc, ns_warm, acc, warm, prefix_warm, prefix_acc in product(namespace, namespace, acc_version, warmup_version, prefix, prefix):
    filename = "bench_%s_%s_%s_%s_%s_%s.art" % (translate_prefix(prefix_acc), acc, ns_acc, translate_prefix(prefix_warm), warm, ns_warm)

    with open(filename, "w+") as f:
        f.write("fn @accelerated_matmul (lzero : Accelerator, a : Tensor[a_element_type], b : Tensor[b_element_type], c : Tensor[c_element_type], d : Tensor[c_element_type]) -> () {\n")
        f.write("%smatrix_multiply_acc_%s [spv_mat_a_datatype, spv_mat_b_datatype, spv_acc_datatype] (lzero, SPVWMMAOperations%s, a, b, c, d);\n" % (prefix_acc, acc, ns_acc))
        f.write("}\n\n")
        f.write("fn @warmup_matmul (lzero : Accelerator, a : Tensor[a_element_type], b : Tensor[b_element_type], c : Tensor[c_element_type], d : Tensor[c_element_type]) -> () {\n")
        if warm == "simple":
            f.write("%smatrix_multiply_acc_simple (lzero, a, b, c, d);\n" % prefix_warm)
        else:
            f.write("%smatrix_multiply_acc_%s [spv_mat_a_datatype, spv_mat_b_datatype, spv_acc_datatype] (lzero, SPVWMMAOperations%s, a, b, c, d);\n" % (prefix_warm, warm, ns_warm))
        f.write("}\n")

for ns_acc, ns_warm, acc, warm, prefix_warm, prefix_acc in product(namespace, namespace, acc_version, warmup_version, prefix, prefix):
    filename = "test_%s_%s_%s_%s_%s_%s.art" % (translate_prefix(prefix_acc), acc, ns_acc, translate_prefix(prefix_warm), warm, ns_warm)

    with open(filename, "w+") as f:
        f.write("fn @accelerated_matmul (lzero : Accelerator, a : Tensor[a_element_type], b : Tensor[b_element_type], c : Tensor[c_element_type], d : Tensor[c_element_type]) -> () {\n")
        f.write("%smatrix_multiply_acc_%s [spv_mat_a_datatype, spv_mat_b_datatype, spv_acc_datatype] (lzero, SPVWMMAOperations%s, a, b, c, d);\n" % (prefix_acc, acc, ns_acc))
        f.write("}\n\n")
        f.write("fn @warmup_matmul (lzero : Accelerator, a : Tensor[a_element_type], b : Tensor[b_element_type], c : Tensor[c_element_type], d : Tensor[c_element_type]) -> () {\n")
        if warm == "simple":
            f.write("%smatrix_multiply_acc_simple (lzero, a, b, c, d);\n" % prefix_warm)
        else:
            f.write("%smatrix_multiply_acc_%s [spv_mat_a_datatype, spv_mat_b_datatype, spv_acc_datatype] (lzero, SPVWMMAOperations%s, a, b, c, d);\n" % (prefix_warm, warm, ns_warm))
        f.write("}\n")
