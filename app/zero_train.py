import argparse

import lightly_train


def main():
  parser = argparse.ArgumentParser(description="ZeroTrain")
  parser.add_argument('--out', help='输出目录', )
  parser.add_argument('--data', help='图像目录')
  parser.add_argument('--model', help='模型路径')
  parser.add_argument('--epochs', help='训练轮数', type=int, default=300)
  parser.add_argument('--batch_size', help='批量大小', type=int, default=128)
  parser.add_argument('--resume', help='恢复训练', type=bool, default=False)

  args = parser.parse_args()
  out = args.out
  data = args.data
  model = args.model
  epochs = args.epochs
  batch_size = args.batch_size
  resume = args.resume
  lightly_train.train(
    out=out,
    data=data,
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    resume=resume
  )


if __name__ == '__main__':
  main()
