# well log classify

## tips

1. 同样的代码，同样的随机数种子，在服务器Titan显卡上和我本地3060跑出来的结果不一样，但是和我笔记本1050ti跑出来的是一样的，是因为架构不一样吗？

## log

* 2023/9/21

  1. 把transformer加回来了，顺便做了windows兼容性测试（主要是文件方面）
  2. 注意，windows需要把dataloader的进程数量置零，是的，Windows不允许多进程加载哦。（是进程不是线程哦）