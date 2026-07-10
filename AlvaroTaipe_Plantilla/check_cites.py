import re, glob
keys = set()
for f in glob.glob('Chapters/*.tex') + glob.glob('Appendix/*.tex') + ['main.tex']:
    with open(f, encoding='utf-8') as fp:
        for m in re.finditer(r'\\cite\{([^}]+)\}', fp.read()):
            for k in m.group(1).split(','):
                keys.add(k.strip())
all_bib = set()
with open('main.bib', encoding='utf-8') as fp:
    for m in re.finditer(r'^@\w+\{([a-zA-Z0-9_]+),', fp.read(), re.M):
        all_bib.add(m.group(1))
print('UNUSED:', sorted(all_bib - keys))
print('MISSING IN BIB (cited but not defined):', sorted(keys - all_bib))
print('USED:', len(keys), 'BIB TOTAL:', len(all_bib))
