import prep
import read
import plt
import sh


a = read.read_dir(dir_path="./data/saudaveis",
                  file_type=read.FileType.DPT, group="saudaveis", color="blue")
b = read.read_dir(dir_path="./data/cardiopatia",
                  file_type=read.FileType.DPT, group="cardiopatia", color="red")

c = prep.group(a, b)
c = prep.golay(c, 0, 0, 1)
c = prep.norm_vec(c)
c = prep.cut(c, 600, 1800)
sh.mplot_peaks(c)
