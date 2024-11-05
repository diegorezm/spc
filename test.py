import csv
import prep
import read
import sh


a = read.read_dir(dir_path="./data/sla/saudaveis",
                  file_type=read.FileType.DPT, group="saudaveis", color="blue")
b = read.read_dir(dir_path="./data/sla/fumantes",
                  file_type=read.FileType.DPT, group="fumantes", color="red")

c = prep.group(a, b)
c = prep.golay(c, 0, 0, 1)
c = prep.norm_vec(c)
c = prep.cut(c, 600, 1800)

fig, peaks = sh.mplot_peaks_fig(c)
fig.savefig('peaks.png')


with open('peaks.csv', 'w') as f:
    field_names = ['grupo', 'x', 'y', 'valor_absorcao']
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    for key, value in peaks.items():
        ag = a.group_ids[0]
        bg = b.group_ids[0]
        group = "saudaveis" if ag == key[2] else "fumantes"

        writer.writerow(
            {'grupo': group, 'x': key[0], 'y': key[1], 'valor_absorcao': value})
