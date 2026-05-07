# fix_fortran.py
with open('code.for.original', 'r') as f:
    lines = f.readlines()

with open('code.for', 'w') as f:
    for line in lines:
        # If line is not empty and doesn't start with a number (statement label)
        # and doesn't already have 6 or more spaces
        if line.strip() and not line[0].isdigit() and not line.startswith('      '):
            # Add 6 spaces to the beginning
            f.write('      ' + line)
        else:
            f.write(line)
