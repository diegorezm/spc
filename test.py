import read
import prep
import pca

a = read.read_dir(dir_path="./data/saudaveis", file_type=read.FileType.DPT, group="saudaveis", color="blue")
b = read.read_dir(dir_path="./data/cardiopatia", file_type=read.FileType.DPT, group="cardiopatia", color="red")

c = prep.group(a,b)
c = prep.norm_vec(c)
c = prep.golay(c,0,0,1)
c = prep.cut(c, 600, 1800)
pca.scores_plot(c,1,2, "PCA")
