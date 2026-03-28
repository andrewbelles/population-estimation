{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    gcc
    gnumake
    gdal 
    proj 
    geos 

    # Python packages from nixpkgs
    python312Packages.numpy
    python312Packages.scipy
    python312Packages.pandas
    python312Packages.scikit-learn
    python312Packages.matplotlib
    python312Packages.seaborn
    python312Packages.pyyaml
    python312Packages.geopandas
    python312Packages.xarray
    python312Packages.rasterio
    python312Packages.pybind11
    python312Packages.fiona
    python312Packages.libpysal
    python312Packages.optuna
    python312Packages.umap-learn

    google-cloud-sdk

    cudaPackages.cudatoolkit  
    cudaPackages.nccl 
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.glibc
    pkgs.zlib
    pkgs.libffi
    pkgs.cudaPackages.cudatoolkit
  ] + ":/run/opengl-driver/lib";

  shellHook = ''
    export PROJECT_ROOT="$(pwd)"
    export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export PATH="$PROJECT_ROOT/scripts:$PATH"

    export PIP_DISABLE_PIP_VERSION_CHECK=1 
    export PIP_PROGRESS_BAR=off

    echo "[NIX-SHELL] PROJECT_ROOT set to: $PROJECT_ROOT" 
    echo "[NIX-SHELL] initializing python environment..."

    if [ ! -d ".venv" ]; then 
      echo "[NIX-SHELL] creating new virtual environment..."
      python -m venv .venv 
    fi 

    if [ -n "$VIRTUAL_ENV" ]; then 
      deactivate || true  
    fi

    source .venv/bin/activate
    echo "[NIX-SHELL] activated virtual environment: $VIRTUAL_ENV"
    
    if [ ! -f ".venv/.bootstrapped" ]; then 
      echo "[NIX-SHELL] installing remaining packages via pip"
      PIP_QUIET="python -m pip -q"

      $PIP_QUIET install --upgrade pip || echo "[pip] upgrade failed" >&2 
      $PIP_QUIET install torch torchvision torchaudio\
        --index-url https://download.pytorch.org/whl/cu124 || 
        echo "[pip] torch install failed" >&2 
      $PIP_QUIET install torch-scatter torch-sparse \
        -f https://data.pyg.ord/whl/torch-2.6.0+cu124.html ||
        echo "[pip] pyg deps failed" >&2 

      $PIP_QUIET install torch-geometric rasterstats pyrosm networkit shap imageio || 
        echo "[pip] extra deps failed" >&2 

      echo "[NIX-SHELL] installing CUDA xgboost"
      $PIP_QUIET install --no-cache-dir 'xgboost>=2.0.0' \
        --config-settings=use_cuda=ON \
        --config-settings=use_nccl=ON || echo "[pip] xgboost install failed" >&2 

      $PIP_QUIET install -e . || echo "[pip] editable install failed" >&2 
      touch .venv/.bootstrapped 
    fi 

    echo "[NIX-SHELL] creating project directories outside git repo"
    mkdir -p data/datasets data/tensors  

    echo "[NIX-SHELL] injecting python development headers"
    PYTHON_INC=$(${pkgs.python312}/bin/python -c "import sysconfig; \
      print(sysconfig.get_paths()['include'])")

    export C_INCLUDE_PATH="$PYTHON_INC:''${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="$PYTHON_INC:''${CPLUS_INCLUDE_PATH:-}"
    export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"

    echo "[NIX-SHELL] Injecting pybind11 include path into .clangd"
    PYBIND11_INC=$(${pkgs.python312}/bin/python -c "import pybind11; \
      print(pybind11.get_include())")
    TORCH_INCS=$(python -c "from torch.utils.cpp_extension import include_paths; \
      print('\n'.join(include_paths()))" 2>/dev/null || true) 

cat > .clangd <<EOF
CompileFlags:
  Add:
    - "-std=c++17"
    - "-DTORCH_EXTENSION_NAME=topo_kernels"
    - "-${pkgs.cudaPackages.cudatoolkit}/include"
    - "-I$PYBIND11_INC"
    - "-I$PYTHON_INC"
EOF
    if [ -n "$TORCH_INCS" ]; then
      while IFS= read -r inc; do 
        echo "    - \"-I$inc\"" >> .clangd
      done <<< "$TORCH_INCS"
    fi 
    '';
}
