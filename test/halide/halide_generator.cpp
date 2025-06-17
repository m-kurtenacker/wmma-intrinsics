#include "Halide.h"

#include <iostream>

using namespace Halide;

/*class MatmulGenerator : public Generator<MatmulGenerator> {
    public:
        Input<Buffer<float, 2>> A{"A"};
        Input<Buffer<float, 2>> B{"B"};
        Input<Buffer<float, 2>> C{"C"};

        Var x, y;

        Output<Buffer<float, 2>> matmul{"matmul"};

        void generate() {
            matmul(x, y) = C(x, y);

            RDom k(0, A.dim(1).extent());
            matmul(x, y) += A(x, k) * B(k, y);
        }
};

HALIDE_REGISTER_GENERATOR(MatmulGenerator, matmul_generator)*/

int main(int argc, char ** argv) {
    ImageParam A(Float(32), 2, "A");
    ImageParam B(Float(32), 2, "B");
    ImageParam C(Float(32), 2, "C");

    Var x("x"), y("y");
    Func matmul;

    matmul(x, y) = C(x, y);

    RDom k(0, A.dim(1).extent(), "k");
    matmul(x, y) += A(x, k) * B(k, y);

    //Schedule
    try {
        matmul.vectorize(y, 8);

        Var x_outer, x_inner, y_outer, y_inner;
        matmul.update().tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
        matmul.update().parallel(y_outer);

        //Var kx("kx");
        //Func intermediate = matmul.update().rfactor({{k, kx}});
        //intermediate.compute_root().update().parallel(kx);

        matmul.compile_to_static_library("matmul", {A, B, C}, "matmul");
    } catch (CompileError e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
