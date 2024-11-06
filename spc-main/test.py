import pca
import prep
import read
import sh


a = read.read_dir(dir_path="./rato_nada", file_type=read.FileType.DPT, group="rato_nada", color="blue")
b = read.read_dir(dir_path="./rato_estresse", file_type=read.FileType.DPT, group="rato_estresse", color="red")




c = prep.group(a, b)
c = prep.norm_vec(c)
c = prep.golay(c, 0, 0, 1)
c = prep.cut(c, 500, 1800)



sh.mplot(c)
prep.std_plot(c)
sh.mplot_peaks(c)
pca.scores_plot(c, 1, 2, "PCA")
pca.loading_plt(c, [1, 2])


#seg_deriv_plot

#c = prep.group(a, b)
#c = prep.norm_vec(c) 
#c = prep.golay(c, 2, 2, 11)
#c = prep.cut(c, 500, 1800)

#sh.mplot(c)
#prep.std_plot(c)
#pca.scores_plot(c, 1, 2, "PCA")
#pca.loading_plt(c, [1, 2])