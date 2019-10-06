'''
transform the feature file from cotype-format to our format
'''

import argparse
import shutil
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform CoType preprocessed data to our format.")
    parser.add_argument('--input', nargs='+', required=True, help='input folder')
    parser.add_argument('--output', nargs='+', required=True, help='output folder')
    args = parser.parse_args()

    assert (len(args.input) == len(args.output))

    for (infile, outfile) in zip(args.input, args.output):
        print(infile)
        with open(os.path.join(infile, 'train_x_new.txt'), 'r') as x_in, \
                open(os.path.join(infile, 'train_y.txt'), 'r') as y_in, \
                open(os.path.join(outfile, 'train.txt'), 'w') as fout:
            insmap = {}
            count = 0
            for line in x_in:
                line = line.split('\t')
                feature = list(map(lambda t: int(t), line[1].split(',')))
                tmp = count
                insmap[count] = str(tmp) + '\t' + str(len(feature)) + '\t' + ' '.join(map(lambda t: str(t), feature))
                count += 1
            count = 0
            for line in y_in:
                line = line.split('\t')
                feature = list(map(lambda t: int(t), line[1].split(',')))
                insmap[count] = insmap[count] + '\t' + str(len(feature)) + '\t' + ' '.join(
                    map(lambda t: str(t), feature))
                count += 1
            for (k, v) in insmap.items():
                fout.write(v + '\n')

        with open(os.path.join(infile, 'test_x.txt'), 'r') as x_in, \
                open(os.path.join(infile, 'test_y.txt'), 'r') as y_in, \
                open(os.path.join(outfile, 'test.txt'), 'w') as fout:
            count = 0
            insmap = {}
            for line in x_in:
                line = line.split('\t')
                feature = list(map(lambda t: int(t), line[1].split(',')))
                tmp = count
                insmap[count] = [0,
                                 str(tmp) + '\t' + str(len(feature)) + '\t' + ' '.join(map(lambda t: str(t), feature))]
                count += 1
            count = 0
            for line in y_in:
                line = line.split('\t')
                if len(line[1]) > 1:
                    feature = list(map(lambda t: int(t), line[1].split(',')))
                    insmap[count] = [1, insmap[count][1] + '\t' + str(len(feature)) + '\t' + ' '.join(
                        map(lambda t: str(t), feature))]
                count += 1
            for (k, v) in insmap.items():
                if v[0] == 1:
                    fout.write(v[1] + '\n')

        # add ins num to top
        with open(os.path.join(outfile, 'train.txt'), 'r') as fin, open(os.path.join(outfile, 'train.data'), 'w') as fout:
            lines = fin.readlines()
            fout.write(str(len(lines)) + '\n')
            fout.write(''.join(lines))
        with open(os.path.join(outfile, 'test.txt'), 'r') as fin, open(os.path.join(outfile, 'test.data'), 'w') as fout:
            lines = fin.readlines()
            fout.write(str(len(lines)) + '\n')
            fout.write(''.join(lines))

        print('Transforming dev set...')
        with open(os.path.join(infile, 'dev_x.txt'), 'r') as x_in, \
                open(os.path.join(infile, 'dev_y.txt'), 'r') as y_in, \
                open(os.path.join(outfile, 'dev.txt'), 'w') as fout:
            count = 0
            insmap = {}
            for line in x_in:
                line = line.split('\t')
                feature = list(map(lambda t: int(t), line[1].split(',')))
                tmp = count
                insmap[count] = [0,
                                 str(tmp) + '\t' + str(len(feature)) + '\t' + ' '.join(map(lambda t: str(t), feature))]
                count += 1
            count = 0
            for line in y_in:
                line = line.split('\t')
                if len(line[1]) > 1:
                    feature = list(map(lambda t: int(t), line[1].split(',')))
                    insmap[count] = [1, insmap[count][1] + '\t' + str(len(feature)) + '\t' + ' '.join(
                        map(lambda t: str(t), feature))]
                count += 1
            for (k, v) in insmap.items():
                if v[0] == 1:
                    fout.write(v[1] + '\n')
        with open(os.path.join(outfile, 'dev.txt'), 'r') as fin, open(os.path.join(outfile, 'dev.data'), 'w') as fout:
            lines = fin.readlines()
            fout.write(str(len(lines)) + '\n')
            fout.write(''.join(lines))

        # copy qa and feature files from original folder
        # shutil.copy(os.path.join(infile, 'feature.txt'), outfile)
        # shutil.copy(os.path.join(infile, 'qa_x_new.txt'), outfile)
        # shutil.copy(os.path.join(infile, 'mention_question.txt'), outfile)
        # shutil.copy(os.path.join(infile, 'type_test.txt'), outfile)
