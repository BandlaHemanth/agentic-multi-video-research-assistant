"""test_codebase_integrity.py — Project-wide integrity checks to prevent regression."""
import os
import sys

def test_no_stale_agent_client_references():
    bad_pattern = "agent.client"
    found_issues = []
    
    # Resolve relative paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    for idx, line in enumerate(lines):
                        stripped = line.strip()
                        # Match actual code usage of agent.client (exclude comments and string literal assertions)
                        if bad_pattern in stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith('//'):
                            found_issues.append(f"{os.path.relpath(path, project_root)}:L{idx+1} - {stripped}")
                except Exception as e:
                    pass
                    
    if found_issues:
        print(f"\n[FAIL] Found {len(found_issues)} obsolete 'agent.client' reference(s):")
        for issue in found_issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("\n[PASS] Project-wide check complete: No obsolete 'agent.client' references found.")
        sys.exit(0)

if __name__ == "__main__":
    test_no_stale_agent_client_references()
