import argparse
import os

from utils import check_and_install, exec_command, exec_python_command, rename_file_overwrite

current_dir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="MagicLabel")
  parser.add_argument('--task', help='任务类型', default="detect")
  parser.add_argument('--framework', help='执行框架', default="ultralytics")
  parser.add_argument('--model', help='模型路径')

  args = parser.parse_args()
  task = args.task
  framework = args.framework
  model = args.model

  base_path = os.path.normpath(os.path.dirname(model))
  model_name = os.path.splitext(os.path.basename(model))[0]
  model_name_ncnn = model_name.replace('-', '_')

  export_torchscript_filename = os.path.join(base_path, "{}.torchscript".format(model_name))
  export_pnnx_pt_filename = os.path.join(base_path, "{}_pnnx.py.pt".format(model_name_ncnn))

  export_pnnx_onnx_filename = os.path.join(base_path, "{}.pnnx.onnx".format(model_name_ncnn))

  export_pnnx_param_filename = os.path.join(base_path, "{}.pnnx.param".format(model_name_ncnn))
  export_pnnx_bin_filename = os.path.join(base_path, "{}.pnnx.bin".format(model_name_ncnn))

  export_ncnn_param_filename = os.path.join(base_path, "{}.ncnn.param".format(model_name_ncnn))
  export_ncnn_bin_filename = os.path.join(base_path, "{}.ncnn.bin".format(model_name_ncnn))

  export_pnnx_py_filename = os.path.join(base_path, "{}_pnnx.py".format(model_name_ncnn))
  export_ncnn_py_filename = os.path.join(base_path, "{}_ncnn.py".format(model_name_ncnn))

  export_pnnx_py_pt_onnx_filename = os.path.join(base_path, "{}_pnnx.py.pnnx.onnx".format(model_name_ncnn))

  export_pnnx_py_param_filename = os.path.join(base_path, "{}_pnnx.py.pnnx.param".format(model_name_ncnn))
  export_pnnx_py_bin_filename = os.path.join(base_path, "{}_pnnx.py.pnnx.bin".format(model_name_ncnn))

  export_pnnx_py_ncnn_param_filename = os.path.join(base_path, "{}_pnnx.py.ncnn.param".format(model_name_ncnn))
  export_pnnx_py_ncnn_bin_filename = os.path.join(base_path, "{}_pnnx.py.ncnn.bin".format(model_name_ncnn))

  export_pnnx_py_py_filename = os.path.join(base_path, "{}_pnnx.py_pnnx.py".format(model_name_ncnn))
  export_ncnn_py_py_filename = os.path.join(base_path, "{}_pnnx.py_ncnn.py".format(model_name_ncnn))

  def print_dest_ncnn_path():
    print("导出ncnn模型成功")
    print("param文件路径：{}".format(export_ncnn_param_filename))
    print("bin文件路径：{}".format(export_ncnn_bin_filename))

  def remove_temp_file():
    remove_files = [export_torchscript_filename, export_pnnx_onnx_filename, export_pnnx_bin_filename,
                    export_pnnx_param_filename, export_pnnx_py_filename, export_ncnn_py_filename]

    for file in remove_files:
      os.remove(file)

    check_remove_files = [export_pnnx_py_param_filename, export_pnnx_py_bin_filename, export_pnnx_py_py_filename,
                          export_ncnn_py_py_filename, export_pnnx_pt_filename, export_pnnx_py_pt_onnx_filename]

    for file in check_remove_files:
      if os.path.exists(file):
        os.remove(file)

  def pnnx_pt_export(*params):
    exec_command("pnnx", export_pnnx_pt_filename, *params)
    rename_file_overwrite(export_pnnx_py_ncnn_param_filename, export_ncnn_param_filename)
    rename_file_overwrite(export_pnnx_py_ncnn_bin_filename, export_ncnn_bin_filename)
    remove_temp_file()
    print_dest_ncnn_path()

  command = []
  if framework == "yolov5":
    export_py_path = os.path.join(current_dir, "export_yolov5.py")
    command.append(export_py_path)
    command.append("--weights")
    command.append(model)
    command.append("--include")
    command.append("torchscript")
  else:
    export_py_path = os.path.join(current_dir, "export.py")
    command.append(export_py_path)
    command.append("export")
    command.append('model="{}"'.format(model))
    command.append('format="{}"'.format("torchscript"))

  check_and_install("pnnx", "ncnn")
  if exec_python_command(*command):
    if exec_command("pnnx", export_torchscript_filename):
      remove_temp_file()
      print_dest_ncnn_path()
      return
    print("导出ncnn模型失败")
  else:
    print("导出ncnn模型失败")

if __name__ == "__main__":
  main()
