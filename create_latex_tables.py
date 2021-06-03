import json
import argparse

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--filename', help='filename to convert to table', type=str)
fn = parser.parse_args().filename
f = open(fn)
results = json.load(f)

target = int(fn[-6])

# print(results)

print('\\begin{table}[h]')
print('\\caption{Results with MNIST with a target label $t = %d$ and backdoor pattern ``X.\'\'}' % target)
print('\\centering')
print('\\begin{tabular}{|c|c|l|l|l|l|l|}')
print('\\hline')
print('\\multicolumn{2}{|c|}{$\\alpha$}                                    & 0.00 & 0.05 & 0.15 & 0.20 & 0.30 \\\\ \\hline')
print('\\multirow{2}{*}{Training $0-1$ Loss}   & Vanilla Training         & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % (results['false']['0.0']['Train']['Binary Loss'], results['false']['0.05']['Train']['Binary Loss'], results['false']['0.15']['Train']['Binary Loss'], results['false']['0.2']['Train']['Binary Loss'], results['false']['0.3']['Train']['Binary Loss']))
print('                                       & PGD-Adversarial Training & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline' % (results['true']['0.0']['Train']['Binary Loss'], results['true']['0.05']['Train']['Binary Loss'], results['true']['0.15']['Train']['Binary Loss'], results['true']['0.2']['Train']['Binary Loss'], results['true']['0.3']['Train']['Binary Loss']))

print('\\multirow{2}{*}{Training Robust Loss}  & Vanilla Training         & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % (results['false']['0.0']['Train']['Robust Loss'], results['false']['0.05']['Train']['Robust Loss'], results['false']['0.15']['Train']['Robust Loss'], results['false']['0.2']['Train']['Robust Loss'], results['false']['0.3']['Train']['Robust Loss']))
print('                                       & PGD-Adversarial Training & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline' % (results['true']['0.0']['Train']['Robust Loss'], results['true']['0.05']['Train']['Robust Loss'], results['true']['0.15']['Train']['Robust Loss'], results['true']['0.2']['Train']['Robust Loss'], results['true']['0.3']['Train']['Robust Loss']))

print('\\multirow{2}{*}{Testing $0-1$ Loss}    & Vanilla Training         & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % (results['false']['0.0']['Test']['Binary Loss'], results['false']['0.05']['Test']['Binary Loss'], results['false']['0.15']['Test']['Binary Loss'], results['false']['0.2']['Test']['Binary Loss'], results['false']['0.3']['Test']['Binary Loss']))
print('                                       & PGD-Adversarial Training & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline' % (results['true']['0.0']['Test']['Binary Loss'], results['true']['0.05']['Test']['Binary Loss'], results['true']['0.15']['Test']['Binary Loss'], results['true']['0.2']['Test']['Binary Loss'], results['true']['0.3']['Test']['Binary Loss']))

print('\\multirow{2}{*}{Testing Robust Loss}   & Vanilla Training         & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % (results['false']['0.0']['Test']['Robust Loss'], results['false']['0.05']['Test']['Robust Loss'], results['false']['0.15']['Test']['Robust Loss'], results['false']['0.2']['Test']['Robust Loss'], results['false']['0.3']['Test']['Robust Loss']))
print('                                       & PGD-Adversarial Training & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline' % (results['true']['0.0']['Test']['Robust Loss'], results['true']['0.05']['Test']['Robust Loss'], results['true']['0.15']['Test']['Robust Loss'], results['true']['0.2']['Test']['Robust Loss'], results['true']['0.3']['Test']['Robust Loss']))

print('\\multirow{2}{*}{Backdoor Success Rate} & Vanilla Training         & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % (results['false']['0.0']['Backdoor Accuracy'], results['false']['0.05']['Backdoor Accuracy'], results['false']['0.15']['Backdoor Accuracy'], results['false']['0.2']['Backdoor Accuracy'], results['false']['0.3']['Backdoor Accuracy']))
print('                                       & PGD-Adversarial Training & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline' % (results['true']['0.0']['Backdoor Accuracy'], results['true']['0.05']['Backdoor Accuracy'], results['true']['0.15']['Backdoor Accuracy'], results['true']['0.2']['Backdoor Accuracy'], results['true']['0.3']['Backdoor Accuracy']))
print('\\end{tabular}')
print('\\end{table}')