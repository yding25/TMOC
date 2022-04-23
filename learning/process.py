import numpy as np

# ------------------------------------------------------------------------
# read data
rawdata = np.loadtxt('learntExperience(feasibility)_full_updated.txt', dtype='float')
fidout = open('learntExperience_simplified.txt', 'w')
processdata1 = np.empty([1, 10])
for line in rawdata:
    if line[0] <= line[1]:
        if line[4] == 0.4 or line[4] == 0.1:
            processdata1 = np.vstack((processdata1, line))
processdata1 = np.delete(processdata1, 0, axis=0)
# print(processdata1)
# ------------------------------------------------------------------------
# remove reduandacy
processdata2 = processdata1[:, 0:5]
record_line_index = []
temp = np.zeros([1, 5])
i = -1
for line in processdata2:
    i = i + 1
    print('line:', line)
    print('temp:', temp)
    if not (temp == line).all(1).any():
        temp = np.vstack((temp, line))
        record_line_index.append(i)
print(record_line_index)
# ------------------------------------------------------------------------
processdata3 = processdata1[record_line_index, :]
print(len(processdata3))
# ------------------------------------------------------------------------
fidout1 = open('learntExperience_simplified_sorted.txt', 'w')
for i in np.linspace(1.0, 5.0, 21):
    for line in processdata3:
        if round(line[0], 1) == round(i, 1):
            fidout1.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]))
fidout1.close()
