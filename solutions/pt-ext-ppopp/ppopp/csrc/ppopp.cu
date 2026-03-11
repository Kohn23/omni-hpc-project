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

// kernels
#include <ppopp.h>

namespace ppopp {
  // TODO: 

// TODO: Wrapper
torch::stable::Tensor svdmatmul_cuda(const torch::stable::Tensor &a,
                                      const torch::stable::Tensor &b) {

  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CUDA);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);



}

// Defines the operators
STABLE_TORCH_LIBRARY(ppopp, m) {
  m.def("svdmatmul(Tensor a, Tensor b) -> Tensor");
}

// Registers CUDA implementations for svdmatmul
STABLE_TORCH_LIBRARY_IMPL(ppopp, CUDA, m) {
  m.impl("svdmatmul", TORCH_BOX(&svdmatmul_cuda));
}

}


