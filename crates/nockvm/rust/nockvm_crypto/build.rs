use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // 获取 Cargo 的 OUT_DIR 环境变量，编译后的库和对象文件将放在这里
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR 环境变量未设置"));

    // 定义 CUDA 源文件目录的相对路径
    // build.rs 位于 official_nockchain_source/crates/nockvm/rust/nockvm_crypto/
    // CUDA 内核位于 official_nockchain_source/cuda_kernels_for_nockvm/
    // 因此，相对路径是 ../../../../cuda_kernels_for_nockvm/
    let base_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR 未设置"));
    let cuda_src_dir = base_dir.join("../../../../cuda_kernels_for_nockvm");

    // 检查 CUDA 源目录是否存在
    if !cuda_src_dir.exists() {
        panic!(
            "CUDA 源目录未找到于: {:?}. \
             请确保 `cuda_kernels_for_nockvm` 目录相对于 `nockvm_crypto` crate 的位置正确。",
            cuda_src_dir
        );
    }

    // 查找 NVCC 编译器。在 cuda:12.0.1-devel-ubuntu20.04 Docker 镜像中，nvcc 应该在 PATH 中。
    let nvcc_path = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    // 列出所有 .cu 文件
    let cu_files: Vec<PathBuf> = fs::read_dir(&cuda_src_dir)
        .unwrap_or_else(|e| panic!("读取 CUDA 源目录 {:?} 失败: {}", cuda_src_dir, e))
        .filter_map(|entry| {
            let entry = entry.expect("读取目录条目失败");
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "cu") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if cu_files.is_empty() {
        panic!("在 {:?} 中未找到任何 .cu 文件", cuda_src_dir);
    }

    let mut object_files = Vec::new(); // 用于存放编译后的对象文件名

    // 编译每个 .cu 文件
    for cu_file in &cu_files {
        let obj_file_name = cu_file.file_stem().unwrap().to_str().unwrap();
        let obj_file_path = out_dir.join(format!("{}.o", obj_file_name));

        // NVCC 命令参数
        // -c: 只编译，不链接
        // -o: 指定输出文件
        // -Xcompiler -fPIC: 生成位置无关代码 (Position Independent Code)，用于静态库
        // -O3: 优化级别
        // -gencode: 指定目标 GPU 架构。CUDA 12.0 支持 sm_86 (Ampere, 如 RTX 30xx) 和 sm_89 (Ada Lovelace, 如 RTX 40xx)
        //           Vast.ai 通常提供这些现代 GPU。
        // -I: 添加头文件搜索路径
        let mut nvcc_cmd = Command::new(&nvcc_path);
        nvcc_cmd
            .arg("-c")
            .arg(cu_file)
            .arg("-o")
            .arg(&obj_file_path)
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-O3")
            .arg("-gencode")
            .arg("arch=compute_86,code=sm_86") // Ampere 架构
            .arg("-gencode")
            .arg("arch=compute_89,code=sm_89") // Ada Lovelace 架构
            .arg("-I")
            .arg(&cuda_src_dir); // 允许 .cu 文件相互包含头文件（如果头文件也在 cuda_src_dir 中）

        // 可选：如果 CUDA toolkit 的 include 目录不在标准路径，可能需要显式添加
        // 例如: .arg("-I").arg("/usr/local/cuda/include")
        // 但在 devel Docker 镜像中，nvcc 通常能自己找到

        let status = nvcc_cmd.status().expect("执行 nvcc 失败");

        if !status.success() {
            panic!("nvcc 编译 {:?} 失败: {}", cu_file, status);
        }
        object_files.push(obj_file_path);

        // 告诉 Cargo 如果 .cu 文件改变，则重新运行此构建脚本
        println!("cargo:rerun-if-changed={}", cu_file.display());
    }

    // 将编译后的对象文件归档为静态库 (libnockchain_gpu_kernels.a)
    // 库名必须与 gpu_ffi.rs 中 #[link(name = "...")] 指定的名称一致
    let lib_name = "nockchain_gpu_kernels";
    let lib_path = out_dir.join(format!("lib{}.a", lib_name));

    // 使用 `ar` 命令创建静态库
    // 如果库已存在，先删除，以确保全新创建
    if lib_path.exists() {
        fs::remove_file(&lib_path).expect("删除已存在的静态库失败");
    }

    let ar_path = env::var("AR").unwrap_or_else(|_| "ar".to_string()); // 通常 `ar` 在 PATH 中
    let mut ar_cmd = Command::new(ar_path);
    ar_cmd.arg("rcs").arg(&lib_path); // r - 插入或替换文件, c - 创建归档, s - 创建索引
    for obj_file in &object_files {
        ar_cmd.arg(obj_file);
    }

    let status = ar_cmd.status().expect("执行 ar 失败");

    if !status.success() {
        panic!("ar 创建静态库失败: {}", status);
    }

    // 指示 Cargo 如何链接
    println!("cargo:rustc-link-search=native={}", out_dir.display()); // 添加 OUT_DIR 到库搜索路径
    println!("cargo:rustc-link-lib=static={}", lib_name);            // 链接我们的静态库

    // 链接 CUDA 运行时库 (cudart)
    // 在 devel Docker 镜像中，这个库应该是可用的
    println!("cargo:rustc-link-lib=dylib=cudart");

    // 如果 CUDA 内核源目录中的头文件 (.h) 发生变化，也需要重新运行构建脚本
    let h_files: Vec<PathBuf> = fs::read_dir(&cuda_src_dir)
        .unwrap_or_else(|e| panic!("再次读取 CUDA 源目录 {:?} 失败: {}", cuda_src_dir, e))
        .filter_map(|entry| {
            let entry = entry.expect("读取目录条目失败");
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "h") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    for h_file in &h_files {
        println!("cargo:rerun-if-changed={}", h_file.display());
    }

    // 如果 build.rs 本身发生变化，也需要重新运行
    println!("cargo:rerun-if-changed=build.rs");
} 