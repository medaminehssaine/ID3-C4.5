#!/usr/bin/env python3
"""
Realistic GitHub Commit Generator
Makes human-like commits to a Git repository
"""

import os
import sys
import random
import subprocess
import datetime
import time
from pathlib import Path
import json
from typing import List, Dict, Optional
import re

class RealisticCommitGenerator:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).absolute()
        self.git_dir = self.repo_path / ".git"
        
        if not self.git_dir.exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")
        
        self.author_name = self.get_git_config("user.name")
        self.author_email = self.get_git_config("user.email")
        
        # Human-like commit patterns
        self.commit_patterns = {
            'morning': (9, 12),    # 9 AM - 12 PM
            'afternoon': (14, 18),  # 2 PM - 6 PM
            'evening': (19, 23),    # 7 PM - 11 PM
            'weekend_morning': (10, 13),  # Weekend mornings
            'weekend_afternoon': (15, 20), # Weekend afternoons
        }
        
        # Realistic commit messages based on your project structure
        self.commit_messages = [
            # C4.5 related
            "Implement gain ratio calculation for C4.5",
            "Add pruning functionality to decision tree",
            "Fix information gain calculation edge case",
            "Optimize tree node structure for better memory usage",
            "Add validation for input data in C4.5 algorithm",
            "Update documentation for gain ratio implementation",
            
            # ID3 related
            "Improve entropy calculation precision",
            "Add support for categorical features in ID3",
            "Fix issue with empty branches in decision tree",
            "Refactor node splitting logic for better readability",
            "Add unit tests for entropy calculations",
            
            # Data processing
            "Enhance data loader to handle missing values",
            "Add data preprocessing pipeline",
            "Fix data normalization bug",
            "Improve CSV parsing performance",
            
            # Visualization
            "Update tree visualization with better colors",
            "Add graphviz export functionality",
            "Fix visualization for deep trees",
            "Improve tree rendering performance",
            
            # General improvements
            "Refactor code structure for better modularity",
            "Update requirements and dependencies",
            "Fix typos in documentation",
            "Add example usage in README",
            "Improve error messages for better debugging",
            "Optimize imports and clean up code",
            "Add type hints for better IDE support",
            "Update .gitignore with common patterns",
            
            # Test related
            "Add integration tests for decision trees",
            "Fix failing test cases",
            "Improve test coverage for core modules",
            "Add test data for edge cases",
            
            # Bug fixes
            "Fix memory leak in tree construction",
            "Resolve issue with recursion depth",
            "Fix accuracy calculation bug",
            "Patch security vulnerability in data loading",
        ]
        
        # File modification patterns based on your project structure
        self.modification_patterns = {
            'src/decision_trees/c45/': [
                'core/gain_ratio.py',
                'core/tree.py',
                'core/pruning.py',
                'core/node.py',
            ],
            'src/decision_trees/id3/': [
                'core/entropy.py',
                'core/tree.py',
                'core/node.py',
            ],
            'src/decision_trees/c45/utils/': [
                'visualization.py',
            ],
            'src/decision_trees/id3/utils/': [
                'visualization.py',
                'validation.py',
            ],
            'examples/': [
                'demo_c45.py',
                'demo_id3.py',
            ],
            'tests/': [
                'test_c45.py',
                'test_id3.py',
            ],
            '': [  # Root files
                'README.md',
                'pyproject.toml',
                '.gitignore',
            ]
        }

    def get_git_config(self, key: str) -> str:
        """Get git configuration value"""
        try:
            result = subprocess.run(
                ['git', 'config', '--get', key],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except:
            return ""

    def get_commit_history(self) -> List[Dict]:
        """Analyze existing commit patterns"""
        cmd = [
            'git', 'log', '--pretty=format:%H|%an|%ae|%ad|%s',
            '--date=iso',
            '--max-count=100'
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.repo_path, 
                                  capture_output=True, text=True)
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 4)
                    if len(parts) == 5:
                        commits.append({
                            'hash': parts[0],
                            'author': parts[1],
                            'email': parts[2],
                            'date': parts[3],
                            'message': parts[4]
                        })
            return commits
        except:
            return []

    def analyze_commit_patterns(self) -> Dict:
        """Analyze when commits usually happen"""
        commits = self.get_commit_history()
        if not commits:
            return {'weekdays': [0, 1, 2, 3, 4], 'hours': [10, 14, 16, 20]}
        
        weekdays = []
        hours = []
        
        for commit in commits:
            try:
                dt = datetime.datetime.fromisoformat(commit['date'].split()[0])
                weekdays.append(dt.weekday())
                hours.append(dt.hour)
            except:
                continue
        
        # Get most common patterns
        from collections import Counter
        common_weekdays = [day for day, _ in Counter(weekdays).most_common(3)]
        common_hours = [hour for hour, _ in Counter(hours).most_common(4)]
        
        return {
            'weekdays': common_weekdays if common_weekdays else [0, 2, 4],
            'hours': common_hours if common_hours else [10, 14, 16, 20]
        }

    def get_realistic_commit_time(self) -> datetime.datetime:
        """Generate a realistic commit timestamp"""
        patterns = self.analyze_commit_patterns()
        
        # Pick a random weekday from common patterns
        weekday = random.choice(patterns['weekdays'])
        
        # Get current date and find the next occurrence of that weekday
        today = datetime.datetime.now()
        days_ahead = weekday - today.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        target_date = today + datetime.timedelta(days=days_ahead)
        
        # Pick a realistic hour
        hour = random.choice(patterns['hours'])
        minute = random.randint(0, 59)
        
        # Create datetime object
        commit_time = target_date.replace(
            hour=hour, 
            minute=minute, 
            second=random.randint(0, 59),
            microsecond=0
        )
        
        # Randomly shift back up to 2 weeks for variety
        if random.random() < 0.3:
            commit_time -= datetime.timedelta(days=random.randint(1, 14))
        
        return commit_time

    def make_realistic_modification(self, file_path: Path) -> bool:
        """Make a realistic modification to a file"""
        if not file_path.exists():
            # Create new file with realistic content
            return self.create_new_file(file_path)
        
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        if not content.strip():
            return self.create_new_file(file_path)
        
        # Different modification strategies based on file type
        if file_path.suffix == '.py':
            return self.modify_python_file(file_path, content)
        elif file_path.suffix == '.md':
            return self.modify_markdown_file(file_path, content)
        elif file_path.name == 'pyproject.toml':
            return self.modify_toml_file(file_path, content)
        else:
            return self.modify_generic_file(file_path, content)

    def modify_python_file(self, file_path: Path, content: str) -> bool:
        """Make realistic Python code modifications"""
        lines = content.split('\n')
        
        modifications = []
        
        # 1. Add a new function (30% chance)
        if random.random() < 0.3 and len(lines) < 200:
            func_template = [
                f"def {self.generate_function_name()}():",
                f'    """{self.generate_docstring()}"""',
                "    try:",
                f"        {self.generate_python_expression()}",
                "    except Exception as e:",
                '        print(f"Error: {e}")',
                "        return None",
                "",
            ]
            insert_pos = random.randint(0, len(lines))
            modifications.append(('insert', insert_pos, func_template))
        
        # 2. Modify existing function (40% chance)
        elif random.random() < 0.4:
            # Find functions in the file
            func_pattern = r'^\s*def\s+(\w+)\s*\('
            func_lines = []
            for i, line in enumerate(lines):
                if re.match(func_pattern, line):
                    func_lines.append(i)
            
            if func_lines:
                func_line = random.choice(func_lines)
                # Add a docstring or modify existing one
                if func_line + 1 < len(lines) and '"""' not in lines[func_line + 1]:
                    docstring = f'    """{self.generate_docstring()}"""'
                    modifications.append(('insert', func_line + 1, [docstring]))
                # Add a line to the function body
                body_line = func_line + random.randint(2, min(10, len(lines) - func_line - 1))
                new_line = f"        {self.generate_python_expression()}"
                modifications.append(('insert', body_line, [new_line]))
        
        # 3. Fix a typo or improve variable name (20% chance)
        if random.random() < 0.2 and len(lines) > 10:
            line_idx = random.randint(0, len(lines) - 1)
            line = lines[line_idx]
            if '=' in line and 'def ' not in line and 'class ' not in line:
                # Improve variable name
                parts = line.split('=')
                if len(parts) == 2:
                    var_part = parts[0].strip()
                    if var_part and len(var_part) < 10 and var_part[0].islower():
                        better_name = self.improve_variable_name(var_part)
                        new_line = line.replace(var_part, better_name, 1)
                        modifications.append(('replace', line_idx, [new_line]))
        
        # 4. Add import if needed (10% chance)
        if random.random() < 0.1:
            imports = ['import numpy as np', 'from typing import List, Dict', 
                      'import matplotlib.pyplot as plt', 'from pathlib import Path']
            import_line = random.choice(imports)
            # Find where imports end
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip() and not (line.startswith('import ') or line.startswith('from ')):
                    import_end = i
                    break
            
            if import_end > 0:
                modifications.append(('insert', import_end, [import_line]))
        
        # Apply modifications
        if modifications:
            # Sort modifications by line number (descending) to avoid index issues
            modifications.sort(key=lambda x: x[1], reverse=True)
            
            for mod_type, line_idx, new_lines in modifications:
                if mod_type == 'insert':
                    lines[line_idx:line_idx] = new_lines
                elif mod_type == 'replace':
                    if line_idx < len(lines):
                        lines[line_idx] = new_lines[0]
            
            file_path.write_text('\n'.join(lines))
            return True
        
        return False

    def generate_function_name(self) -> str:
        """Generate realistic function names"""
        prefixes = ['calculate_', 'get_', 'process_', 'validate_', 'update_', 
                   'generate_', 'compute_', 'analyze_', 'predict_', 'train_']
        stems = ['entropy', 'gain', 'ratio', 'tree', 'node', 'split', 
                'data', 'feature', 'accuracy', 'model', 'dataset']
        return random.choice(prefixes) + random.choice(stems)

    def generate_docstring(self) -> str:
        """Generate realistic docstrings"""
        templates = [
            "Calculate {} for decision tree.",
            "Process {} data.",
            "Validate {} parameters.",
            "Generate {} visualization.",
            "Compute {} metric.",
            "Update {} configuration.",
            "Train {} model.",
            "Predict {} values.",
        ]
        subjects = ['information gain', 'entropy', 'gain ratio', 'tree node', 
                   'data split', 'feature importance', 'accuracy score', 'model performance']
        return random.choice(templates).format(random.choice(subjects))

    def generate_python_expression(self) -> str:
        """Generate realistic Python code expressions"""
        expressions = [
            "result = sum(x for x in values if x > threshold)",
            "accuracy = correct / total if total > 0 else 0.0",
            "prediction = model.predict(features.reshape(1, -1))",
            "entropy_val = -sum(p * math.log2(p) for p in probabilities if p > 0)",
            "gain = parent_entropy - weighted_child_entropy",
            "tree_depth = max(node.depth for node in nodes)",
            "feature_importance = calculate_importance(features, target)",
            "normalized_data = (data - data.mean()) / data.std()",
            "confusion_matrix = compute_confusion(predictions, labels)",
            "best_split = find_best_split(features, target, criterion)",
        ]
        return random.choice(expressions)

    def improve_variable_name(self, old_name: str) -> str:
        """Improve a variable name to be more descriptive"""
        improvements = {
            'x': 'value', 'y': 'target', 'z': 'result',
            'i': 'index', 'j': 'counter', 'k': 'iteration',
            'n': 'count', 'm': 'size', 'p': 'probability',
            'd': 'data', 'f': 'feature', 't': 'threshold',
            'a': 'array', 'b': 'buffer', 'c': 'column',
        }
        return improvements.get(old_name, old_name + '_val')

    def modify_markdown_file(self, file_path: Path, content: str) -> bool:
        """Make realistic markdown modifications"""
        lines = content.split('\n')
        
        # Add a new section or update existing one
        sections = [
            "## Performance Improvements",
            "## Bug Fixes", 
            "## New Features",
            "## Usage Examples",
            "## Installation Notes",
            "## Known Issues",
            "## Future Work",
        ]
        
        new_section = random.choice(sections)
        new_content = [
            f"\n{new_section}",
            "",
            f"- {self.generate_bullet_point()}",
            f"- {self.generate_bullet_point()}",
            ""
        ]
        
        # Insert at random position
        insert_pos = random.randint(max(0, len(lines) - 10), len(lines))
        lines[insert_pos:insert_pos] = new_content
        
        file_path.write_text('\n'.join(lines))
        return True

    def generate_bullet_point(self) -> str:
        """Generate realistic bullet points for documentation"""
        points = [
            "Improved accuracy by 2% with better pruning",
            "Added support for categorical variables",
            "Fixed memory leak in large datasets",
            "Enhanced visualization with color coding",
            "Reduced training time by 15%",
            "Added unit tests for edge cases",
            "Updated documentation with examples",
            "Improved error handling for invalid input",
            "Optimized algorithm for better performance",
            "Added parallel processing support",
        ]
        return random.choice(points)

    def modify_toml_file(self, file_path: Path, content: str) -> bool:
        """Make realistic TOML file modifications"""
        lines = content.split('\n')
        
        # Add or update a dependency
        dependencies = [
            'numpy = ">=1.21.0"',
            'matplotlib = ">=3.5.0"',
            'scikit-learn = ">=1.0.0"',
            'pandas = ">=1.3.0"',
            'graphviz = ">=0.20.0"',
            'pytest = ">=7.0.0"',
        ]
        
        new_dep = random.choice(dependencies)
        dep_name = new_dep.split('=')[0].strip()
        
        # Check if dependency already exists
        for i, line in enumerate(lines):
            if dep_name in line and '=' in line:
                # Update version
                lines[i] = new_dep
                break
        else:
            # Add new dependency
            # Find [tool.poetry.dependencies] section
            for i, line in enumerate(lines):
                if '[tool.poetry.dependencies]' in line:
                    insert_pos = i + 1
                    while insert_pos < len(lines) and lines[insert_pos].strip():
                        insert_pos += 1
                    lines.insert(insert_pos, new_dep)
                    break
        
        file_path.write_text('\n'.join(lines))
        return True

    def modify_generic_file(self, file_path: Path, content: str) -> bool:
        """Make modifications to generic files"""
        lines = content.split('\n')
        
        if len(lines) < 5:
            # Add some content
            templates = [
                "# Configuration file\n# Generated automatically\n\n",
                "// Auto-generated file\n// Do not edit manually\n\n",
                "/* Configuration settings */\n/* Updated automatically */\n\n",
            ]
            new_content = random.choice(templates) + content
            file_path.write_text(new_content)
            return True
        
        # Modify a random line
        line_idx = random.randint(0, len(lines) - 1)
        if lines[line_idx].strip():
            # Add a comment or modify slightly
            if random.random() < 0.5:
                lines[line_idx] = f"{lines[line_idx]}  # updated"
            else:
                # Replace common patterns
                replacements = [
                    ('true', 'false'),
                    ('false', 'true'),
                    ('yes', 'no'),
                    ('no', 'yes'),
                    ('enable', 'disable'),
                    ('disable', 'enable'),
                    ('1', '2'),
                    ('0', '1'),
                ]
                for old, new in replacements:
                    if old in lines[line_idx].lower():
                        lines[line_idx] = lines[line_idx].lower().replace(old, new)
                        break
        
        file_path.write_text('\n'.join(lines))
        return True

    def create_new_file(self, file_path: Path) -> bool:
        """Create a new file with realistic content"""
        if file_path.suffix == '.py':
            content = self.generate_new_python_file()
        elif file_path.suffix == '.md':
            content = self.generate_new_markdown_file()
        elif file_path.suffix == '.txt':
            content = self.generate_new_text_file()
        else:
            content = f"# {file_path.name}\n# Created automatically\n\n"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return True

    def generate_new_python_file(self) -> str:
        """Generate a new Python file with realistic content"""
        imports = [
            "import numpy as np\nimport matplotlib.pyplot as plt\nfrom typing import List, Dict, Optional\n\n",
            "from pathlib import Path\nimport json\nimport warnings\nwarnings.filterwarnings('ignore')\n\n",
            "import math\nfrom collections import defaultdict\nfrom dataclasses import dataclass\n\n",
        ]
        
        class_template = '''
class {class_name}:
    """{docstring}"""
    
    def __init__(self{init_params}):
        {init_body}
    
    def {method_name}(self{method_params}):
        """{method_docstring}"""
        {method_body}
        return {return_value}
'''

        class_names = ['DataProcessor', 'ModelValidator', 'FeatureEngineer', 
                      'ResultAnalyzer', 'VisualizationHelper', 'PerformanceMonitor']
        method_names = ['process', 'validate', 'analyze', 'compute', 'generate', 'evaluate']
        
        init_body = [
            "self.data = data\n        self.config = config or {}",
            "self.features = features\n        self.target = target",
            "self.model = model\n        self.params = params",
            "self.input_path = Path(input_path)\n        self.output_path = Path(output_path)",
        ]
        
        method_body = [
            "result = sum(item for item in data if item > threshold)",
            "accuracy = correct_predictions / total_samples",
            "predictions = [self.predict(x) for x in test_data]",
            "features = self.extract_features(raw_data)",
            "visualization = self.create_plot(data, labels)",
        ]
        
        return_value = [
            "result", "accuracy", "predictions", "features", "visualization", "None"
        ]
        
        content = random.choice(imports)
        content += class_template.format(
            class_name=random.choice(class_names),
            docstring=f"Handle {random.choice(['data processing', 'model validation', 'feature engineering'])}",
            init_params=random.choice([", data=None", ", features=None, target=None", ", config=None"]),
            init_body=random.choice(init_body),
            method_name=random.choice(method_names),
            method_params=random.choice([", threshold=0.5", ", test_data=None", ", raw_data=None"]),
            method_docstring=f"Process {random.choice(['input data', 'features', 'predictions'])}",
            method_body=random.choice(method_body),
            return_value=random.choice(return_value)
        )
        
        return content

    def select_files_to_modify(self, count: int = 3) -> List[Path]:
        """Select random files to modify based on realistic patterns"""
        all_files = []
        
        # Get all files from modification patterns
        for base_dir, file_patterns in self.modification_patterns.items():
            base_path = self.repo_path / base_dir
            if base_path.exists():
                for pattern in file_patterns:
                    file_path = base_path / pattern
                    if file_path.exists():
                        all_files.append(file_path)
        
        # Also include some random existing files
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory and __pycache__
            if '.git' in root or '__pycache__' in root:
                continue
            
            for file in files:
                if file.endswith(('.py', '.md', '.txt', '.toml', '.yml', '.yaml')):
                    file_path = Path(root) / file
                    if file_path not in all_files:
                        all_files.append(file_path)
        
        if not all_files:
            # Fallback to creating new files
            return []
        
        # Select random files, but prioritize based on patterns
        selected = []
        for _ in range(min(count, len(all_files))):
            # Weight files by directory (modify core files more often)
            weights = []
            for file in all_files:
                rel_path = str(file.relative_to(self.repo_path))
                if 'core/' in rel_path:
                    weights.append(3.0)
                elif 'src/' in rel_path:
                    weights.append(2.0)
                elif 'tests/' in rel_path or 'examples/' in rel_path:
                    weights.append(1.5)
                else:
                    weights.append(1.0)
            
            # Random selection with weights
            selected_file = random.choices(all_files, weights=weights, k=1)[0]
            selected.append(selected_file)
            all_files.remove(selected_file)
        
        return selected

    def make_commit(self, commit_time: datetime.datetime, 
                   commit_message: str, modifications: List[str]) -> bool:
        """Make a commit with the given timestamp and message"""
        try:
            # Stage changes
            subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
            
            # Set commit timestamp and author
            env = os.environ.copy()
            env['GIT_AUTHOR_DATE'] = commit_time.isoformat()
            env['GIT_COMMITTER_DATE'] = commit_time.isoformat()
            
            # Make commit
            result = subprocess.run(
                ['git', 'commit', '-m', commit_message, '--no-verify'],
                cwd=self.repo_path,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✓ Committed: {commit_message}")
                print(f"  Time: {commit_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Modified: {len(modifications)} files")
                return True
            else:
                print(f"✗ Commit failed: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Error during commit: {e}")
            return False

    def run(self, num_commits: int = 10, push: bool = True):
        """Run the commit generator"""
        print(f"📊 Analyzing commit patterns in {self.repo_path}")
        patterns = self.analyze_commit_patterns()
        print(f"  Common commit days: {[self.weekday_name(d) for d in patterns['weekdays']]}")
        print(f"  Common commit hours: {patterns['hours']}")
        
        print(f"\n🎯 Generating {num_commits} realistic commits...")
        print("=" * 50)
        
        successful_commits = 0
        
        for i in range(num_commits):
            print(f"\n📝 Commit {i + 1}/{num_commits}")
            
            # Generate realistic commit time
            commit_time = self.get_realistic_commit_time()
            
            # Select files to modify
            files_to_modify = self.select_files_to_modify(
                count=random.randint(1, 4)
            )
            
            if not files_to_modify:
                # Create a new file
                new_file_dir = random.choice([
                    self.repo_path / 'src' / 'decision_trees' / 'c45' / 'utils',
                    self.repo_path / 'src' / 'decision_trees' / 'id3' / 'utils',
                    self.repo_path / 'tests',
                    self.repo_path / 'examples',
                ])
                new_file_dir.mkdir(parents=True, exist_ok=True)
                
                file_types = ['.py', '.md', '.txt']
                file_name = f"{self.generate_function_name()}{random.choice(file_types)}"
                files_to_modify = [new_file_dir / file_name]
            
            modifications = []
            for file_path in files_to_modify:
                try:
                    if self.make_realistic_modification(file_path):
                        rel_path = file_path.relative_to(self.repo_path)
                        modifications.append(str(rel_path))
                except Exception as e:
                    print(f"  ⚠️  Failed to modify {file_path}: {e}")
            
            if not modifications:
                print("  ⚠️  No modifications made, skipping...")
                continue
            
            # Select commit message
            commit_message = random.choice(self.commit_messages)
            
            # Make the commit
            if self.make_commit(commit_time, commit_message, modifications):
                successful_commits += 1
            
            # Random delay between commits (simulate human work patterns)
            if i < num_commits - 1:
                delay = random.randint(30, 300)  # 30 seconds to 5 minutes
                print(f"  ⏳ Next commit in {delay} seconds...")
                time.sleep(delay)
        
        print("\n" + "=" * 50)
        print(f"✅ Completed: {successful_commits}/{num_commits} successful commits")
        
        if successful_commits > 0 and push:
            print("\n🚀 Pushing to remote repository...")
            try:
                subprocess.run(['git', 'push', 'origin', 'main'], 
                             cwd=self.repo_path, check=True)
                print("✅ Successfully pushed to GitHub!")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Push failed: {e}")
                print("   You can push manually with: git push origin main")

    def weekday_name(self, weekday_num: int) -> str:
        """Convert weekday number to name"""
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                   'Friday', 'Saturday', 'Sunday']
        return weekdays[weekday_num]

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate realistic Git commits'
    )
    parser.add_argument(
        '--repo', '-r', 
        default='.',
        help='Path to git repository (default: current directory)'
    )
    parser.add_argument(
        '--commits', '-c',
        type=int,
        default=7,
        help='Number of commits to generate (default: 7)'
    )
    parser.add_argument(
        '--no-push',
        action='store_true',
        help='Do not push to remote repository'
    )
    
    args = parser.parse_args()
    
    try:
        generator = RealisticCommitGenerator(args.repo)
        generator.run(
            num_commits=args.commits,
            push=not args.no_push
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
