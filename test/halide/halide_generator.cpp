#include "Halide.h"

#include <iostream>

using namespace Halide;

class MatmulGenerator : public Halide::Generator<MatmulGenerator> {
    private:
        Var x{"x"}, y{"y"};
        //Var x_outer{"xo"}, x_inner{"xi"}, y_outer{"yo"}, y_inner{"yi"};
        Var block{"gpu_block"}, thread{"gpu_thread"}, x_tile{"x_tile"}, y_tile{"y_tile"}, iteration{"iteration"};
        Func matmul{"matmul"};
    public:
        Input<Buffer<float, 2>> A{"A"};
        Input<Buffer<float, 2>> B{"B"};
        Input<Buffer<float, 2>> C{"C"};

        Output<Buffer<float, 2>> D{"D"};

        void generate() {
            matmul(x, y) = C(x, y);

            RDom k(0, A.dim(1).extent(), "k");
            matmul(x, y) += A(x, k) * B(k, y);

            D(x, y) = matmul(x, y);
        }

        void schedule() {
            const Target& target = context().target();

            if (using_autoscheduler()) {
                A.set_estimates({{0, 2048}, {1, 2048}});
                B.set_estimates({{0, 2048}, {1, 2048}});
                C.set_estimates({{0, 2048}, {1, 2048}});
                D.set_estimates({{0, 2048}, {1, 2048}});
            } else if (target.has_gpu_feature()) {
                matmul.compute_root();

                matmul.split(y, block, thread, 32);
                matmul.vectorize(x, target_natural_vector_size(Float(32)));

                matmul.update().tile(x, y, x, y, x_tile, thread, target_natural_vector_size(Float(32)), 32);
                matmul.update().vectorize(x_tile);

                matmul.update().fuse(x, y, block);
                matmul.update().split(block, iteration, block, 20, TailStrategy::Predicate);
                matmul.update().reorder(iteration, block);

                matmul.gpu(block, thread);
                matmul.update().gpu(block, thread);

                matmul.print_loop_nest();
            } else {
                matmul.compute_root();

                matmul.vectorize(x, target_natural_vector_size(Float(32)));

                //Var kx("kx");
                //Func intermediate = matmul.update().rfactor({{k, kx}});
                //intermediate.update().vectorize(kx, 4);

                matmul.update().parallel(y);
                matmul.update().vectorize(x, target_natural_vector_size(Float(32)));
            }
        }
};

HALIDE_REGISTER_GENERATOR(MatmulGenerator, matmul_generator);
