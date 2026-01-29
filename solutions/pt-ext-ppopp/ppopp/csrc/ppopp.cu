#include <Python.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>


extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the STABLE_TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}


namespace pt_ext_ppopp {
// TODO: Implement the CUDA kernels and functions here



// Defines the operators
// TODO
STABLE_TORCH_LIBRARY(extension_cpp_stable, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
// TODO
STABLE_TORCH_LIBRARY_IMPL(extension_cpp_stable, CUDA, m) {
  m.impl("mymuladd", TORCH_BOX(&mymuladd_cuda));
  m.impl("mymul", TORCH_BOX(&mymul_cuda));
  m.impl("myadd_out", TORCH_BOX(&myadd_out_cuda));
}

}


