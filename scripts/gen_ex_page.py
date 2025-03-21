"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent
src = root / "examples"
doc_path = Path('user_guide/examples.md')

"""
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: spacephyml.{ident}")
        """

with mkdocs_gen_files.open(doc_path, "w") as md_fd:
    md_fd.write('# Examples\n')
    for path in sorted(src.rglob("*.py")):
        print(path)
        data = {
            'head': [],
            'code': []
        }
        step = 'head'
        with open(path, 'r') as py_fd:
            for i, line in enumerate(py_fd.readlines()):
                if i == 0 and '"""' in line:
                    continue
                elif i == 0 and '"""' not in line:
                    step = 'code'
                    continue
                elif '"""' in line and step == 'head':
                    step = 'code'
                    continue

                data[step].append(line)

        src_path = path.relative_to(src)
        md_fd.write(f'## {src_path}\n')
        for line in data['head']:
            if line[0] == '#':
                line = '##' + line

            md_fd.write(line)

        md_fd.write('\n')
        md_fd.write('```{python}\n')
        for line in data['code']:
            md_fd.write(line)
        md_fd.write('```\n')


    #mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
