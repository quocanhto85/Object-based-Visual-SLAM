#!/usr/bin/env python3
import sys

def fix_trajectory_format(input_file, output_file):
    fixed_lines = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            values = line.split()
            
            if len(values) != 12:
                if len(values) > 12:
                    values = values[:12]
                elif len(values) == 16:
                    values = values[:12]
                else:
                    print(f"Warning: Line {line_num} has {len(values)} values, skipping")
                    continue
            
            try:
                values_float = [float(v) for v in values]
                values_str = [f"{v:.6e}" for v in values_float]
                fixed_lines.append(' '.join(values_str))
            except ValueError:
                print(f"Error on line {line_num}, skipping")
                continue
    
    with open(output_file, 'w') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    return len(fixed_lines)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 fix_trajectory.py <input> <output>")
        sys.exit(1)
    
    num = fix_trajectory_format(sys.argv[1], sys.argv[2])
    print(f"Fixed {num} poses")
