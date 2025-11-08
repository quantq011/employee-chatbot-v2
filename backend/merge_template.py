#!/usr/bin/env python3
"""
Template Merger for Employee Onboarding
Merges role, region, and phase templates with project-specific overrides
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from copy import deepcopy
 
 
class TemplateMerger:
    def __init__(self, base_path: str = "documents/onboarding"):
        self.base_path = Path(base_path)
        self.templates_path = self.base_path / "templates"
        self.projects_path = self.base_path / "projects"
       
    def load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file and return as dictionary"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}")
            return {}
   
    def save_json(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save dictionary as JSON file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error: Failed to save {file_path}: {e}")
            return False
   
    def deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence
        Special handling for lists with 'additional_' prefix
        """
        result = deepcopy(base)
       
        for key, value in override.items():
            if key in result:
                # If both are dicts, recursively merge
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self.deep_merge(result[key], value)
                # If key starts with 'additional_', append to list
                elif key.startswith('additional_') and isinstance(value, list):
                    base_key = key.replace('additional_', '')
                    if base_key in result and isinstance(result[base_key], list):
                        result[base_key].extend(value)
                    else:
                        result[key] = value
                # Otherwise, override completely replaces base
                else:
                    result[key] = deepcopy(value)
            else:
                result[key] = deepcopy(value)
       
        return result
   
    def load_all_templates(self, role: str = None, region: str = None,
                          phases: List[str] = None) -> Dict[str, Any]:
        """
        Load and combine all template files
       
        Args:
            role: Role template name (e.g., 'backend', 'qa', 'devops')
            region: Region template name (e.g., 'EU', 'US', 'APAC')
            phases: List of phase template names (e.g., ['day0', 'day1', 'week1'])
        """
        merged = {}
       
        # Load role template
        if role:
            role_file = self.templates_path / "role" / f"{role}.json"
            role_data = self.load_json(role_file)
            if role_data:
                merged = self.deep_merge(merged, {"role": role_data})
       
        # Load region template
        if region:
            region_file = self.templates_path / "region" / f"{region}.json"
            region_data = self.load_json(region_file)
            if region_data:
                merged = self.deep_merge(merged, {"region": region_data})
       
        # Load phase templates
        if phases:
            phases_data = {}
            for phase in phases:
                phase_file = self.templates_path / "phase" / f"{phase}.json"
                phase_data = self.load_json(phase_file)
                if phase_data:
                    phases_data[phase] = phase_data
            if phases_data:
                merged["phases"] = phases_data
       
        return merged

    def import_project_files(self, project_name: str) -> Dict[str, Any]:
        """
        Import project definition files from a sibling `documents/projects/{project_name}`
        and write a compatible `overrides.json` into the merger projects folder.

        Supported source files (all optional):
          - project.json              -> becomes project_specific and basic metadata
          - role/<role>.json          -> role overrides (primary role chosen automatically)
          - region/<region>.json      -> region-specific overrides (not required)
          - phase/<phase>.json        -> phase-specific overrides

        Assumptions:
          - If multiple role files exist, the first is treated as the primary role for template selection.
            All role files will be included under `project_specific.roles` for reference.
        - The source folder is resolved as `self.base_path.parent / 'projects' / project_name` so
          placing files in `documents/projects/{project_name}` (relative to repo root) will be found
          when the merger's base_path is `documents/onboarding` (the default).
        """
        src_root = self.base_path.parent / 'projects' / project_name
        if not src_root.exists():
            print(f"No source project folder found at: {src_root}")
            return {}

        overrides: Dict[str, Any] = {}

        # Load project.json if present
        project_file = src_root / 'project.json'
        project_data = {}
        if project_file.exists():
            project_data = self.load_json(project_file)
            overrides.setdefault('project_specific', {})
            overrides['project_specific'].update(project_data)

        # Load role files
        role_dir = src_root / 'role'
        role_files = []
        if role_dir.exists():
            for f in role_dir.glob('*.json'):
                role_files.append(f)

        primary_role = None
        if role_files:
            # Choose the first role file as primary for template selection
            primary_role = role_files[0].stem
            overrides['role'] = primary_role
            # Load primary role as role_overrides
            primary_role_data = self.load_json(role_files[0])
            if primary_role_data:
                overrides['role_overrides'] = primary_role_data

            # Add a list of available roles to project_specific for reference
            overrides.setdefault('project_specific', {})
            overrides['project_specific'].setdefault('roles', [])
            for rf in role_files:
                data = self.load_json(rf)
                overrides['project_specific']['roles'].append({
                    'role': rf.stem,
                    'data': data
                })

        # Load region override if present
        region_dir = src_root / 'region'
        if region_dir.exists():
            # If a single region file exists, pick it
            reg_files = list(region_dir.glob('*.json'))
            if reg_files:
                region_name = reg_files[0].stem
                overrides['region'] = region_name
                region_data = self.load_json(reg_files[0])
                if region_data:
                    overrides['region_overrides'] = region_data

        # Load phase overrides
        phase_dir = src_root / 'phase'
        if phase_dir.exists():
            phase_overrides = {}
            for pf in phase_dir.glob('*.json'):
                phase_overrides[pf.stem] = self.load_json(pf)
            if phase_overrides:
                overrides['phase_overrides'] = phase_overrides

        # Copy top-level metadata like template_name, phases, region if provided in project.json
        if project_data:
            if 'template_name' in project_data:
                overrides['template_name'] = project_data['template_name']
            if 'region' in project_data:
                overrides['region'] = project_data['region']
            if 'phases' in project_data:
                overrides['phases'] = project_data['phases']

        # Ensure the destination project folder exists and save overrides.json
        dest_folder = self.projects_path / project_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        overrides_file = dest_folder / 'overrides.json'
        if self.save_json(overrides, overrides_file):
            print(f"✓ Imported project files from {src_root} -> {overrides_file}")
            return overrides

        print(f"✗ Failed to write overrides.json for project {project_name}")
        return {}
   
    def merge_with_overrides(self, template_name: str, project_name: str,
                           output_file: str = None) -> Dict[str, Any]:
        """
        Main merge function: combines templates with project overrides
       
        Args:
            template_name: Name of the template configuration (stored in overrides.json)
            project_name: Name of the project (folder in projects/)
            output_file: Optional custom output file path
        """
        from datetime import datetime
       
        # Load project overrides; if missing attempt to import from documents/projects
        override_file = self.projects_path / project_name / "overrides.json"
        overrides = self.load_json(override_file)

        if not overrides:
            print(f"overrides.json not found for '{project_name}', attempting to import from documents/projects/{project_name}")
            overrides = self.import_project_files(project_name)
            if not overrides:
                print(f"Error: No overrides found or imported for project '{project_name}'")
                return {}
       
        # Template name is informational only - we use role and region from overrides
        stored_template = overrides.get('template_name', 'unknown')
        print(f"Note: Project overrides specify template_name='{stored_template}', using for merge")
       
        # Determine which templates to load from overrides
        # Default to backend, US, and new phase structure
        role = overrides.get('role', 'backend')
        region_name = overrides.get('region', 'US')
        phases = overrides.get('phases', ['first-3-day', '2-day-after', 'week-02', 'week-03'])
       
        # Load base templates
        print(f"Loading templates: role={role}, region={region_name}, phases={phases}")
        base_templates = self.load_all_templates(role=role, region=region_name, phases=phases)
       
        # Build the new structure
        merged = {
            "metadata": {
                "project_id": project_name,
                "region": region_name,
                "source": f"project:{project_name}",
                "version": datetime.now().strftime("%Y-%m-%d"),
                "template": template_name,
                "generated_at": datetime.now().isoformat()
            },
            "overrides": {}
        }
       
        # Add role information to overrides
        if 'role' in base_templates:
            role_data = base_templates['role']
            # Apply role overrides if exists
            if 'role_overrides' in overrides:
                role_data = self.deep_merge(role_data, overrides['role_overrides'])
            merged['overrides']['role'] = role_data
       
        # Add region information to overrides
        if 'region' in base_templates:
            region_data = base_templates['region']
            # Apply region overrides if exists
            if 'region_overrides' in overrides:
                region_data = self.deep_merge(region_data, overrides['region_overrides'])
            merged['overrides']['region'] = region_data
       
        # Add phases to overrides
        if 'phases' in base_templates:
            phases_data = {}
            for phase_name, phase_content in base_templates['phases'].items():
                # Apply phase overrides if exists
                if 'phase_overrides' in overrides and phase_name in overrides['phase_overrides']:
                    phase_content = self.deep_merge(phase_content, overrides['phase_overrides'][phase_name])
                phases_data[phase_name] = phase_content
            merged['overrides']['phases'] = phases_data
       
        # Add project-specific data to overrides
        if 'project_specific' in overrides:
            merged['overrides']['project_specific'] = overrides['project_specific']
       
        # Save merged result
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.projects_path / project_name / "merged_config.json"
       
        if self.save_json(merged, output_path):
            print(f"\n✓ Successfully merged template '{template_name}' for project '{project_name}'")
            print(f"✓ Output saved to: {output_path}")
       
        return merged
   
    def merge_project_template(self, project_name: str, template_name: str = None,
                              output_file: str = None, merge_sections: List[str] = None) -> Dict[str, Any]:
        """
        Flexible merge function with selective section merging
       
        Args:
            project_name: Name of the project (folder in projects/)
            template_name: Name of the template configuration (optional, uses overrides.json if not provided)
            output_file: Optional custom output file path
            merge_sections: List of sections to merge. Options: ['info', 'region', 'role', 'phases', 'project_specific']
                          If None or contains 'all', merges all sections (default)
       
        Merge sections options:
            - None or ['all']: Merge everything (default)
            - ['info']: Merge only project info (communication tools, management tools)
            - ['region']: Merge only region templates and overrides
            - ['role']: Merge only role templates and overrides
            - ['phases']: Merge only phase templates and overrides
            - ['project_specific']: Merge only project-specific data (repos, contacts)
            - Multiple: e.g., ['role', 'phases'] - merge only specified sections
       
        Examples:
            # Merge everything
            merger.merge_project_template("AC1")
           
            # Merge only project info
            merger.merge_project_template("AC1", merge_sections=['info'])
           
            # Merge role and phases only
            merger.merge_project_template("AC1", merge_sections=['role', 'phases'])
        """
        from datetime import datetime
       
        # Load project overrides; attempt import if missing
        override_file = self.projects_path / project_name / "overrides.json"
        overrides = self.load_json(override_file)
        if not overrides:
            print(f"overrides.json not found for '{project_name}', attempting to import from documents/projects/{project_name}")
            overrides = self.import_project_files(project_name)
            if not overrides:
                print(f"Error: No overrides found or imported for project '{project_name}'")
                return {}
       
        # Extract configuration from overrides
        stored_template = overrides.get('template_name', 'unknown')
        role = overrides.get('role', 'backend')
        region_name = overrides.get('region', 'US')
        phases = overrides.get('phases', ['first-3-day', '2-day-after', 'week-02', 'week-03'])
       
        # Determine which sections to merge
        if merge_sections is None or 'all' in (merge_sections or []):
            merge_sections = ['info', 'region', 'role', 'phases', 'project_specific']
       
        print(f"\n{'='*60}")
        print(f"SELECTIVE MERGE CONFIGURATION")
        print(f"{'='*60}")
        print(f"Project: {project_name}")
        print(f"Template: {template_name or stored_template}")
        print(f"Sections: {', '.join(merge_sections)}")
        print(f"{'='*60}\n")
       
        # Load base templates only if needed
        need_templates = any(s in merge_sections for s in ['role', 'region', 'phases'])
        base_templates = {}
        if need_templates:
            print(f"Loading templates: role={role}, region={region_name}, phases={phases}")
            base_templates = self.load_all_templates(role=role, region=region_name, phases=phases)
       
        # Build the result structure
        merged = {
            "metadata": {
                "project_id": project_name,
                "region": region_name,
                "source": f"project:{project_name}",
                "version": datetime.now().strftime("%Y-%m-%d"),
                "template": template_name or stored_template,
                "generated_at": datetime.now().isoformat(),
                "merged_sections": merge_sections
            },
            "overrides": {}
        }
       
        # Merge project info (communication and management tools)
        if 'info' in merge_sections:
            if 'project_specific' in overrides and 'project_info' in overrides['project_specific']:
                merged['overrides']['project_info'] = overrides['project_specific']['project_info']
                print(f"✓ Merged project info (communication & management tools)")
       
        # Merge role
        if 'role' in merge_sections and 'role' in base_templates:
            role_data = base_templates['role']
            if 'role_overrides' in overrides:
                role_data = self.deep_merge(role_data, overrides['role_overrides'])
            merged['overrides']['role'] = role_data
            print(f"✓ Merged role: {role}")
       
        # Merge region
        if 'region' in merge_sections and 'region' in base_templates:
            region_data = base_templates['region']
            if 'region_overrides' in overrides:
                region_data = self.deep_merge(region_data, overrides['region_overrides'])
            merged['overrides']['region'] = region_data
            print(f"✓ Merged region: {region_name}")
       
        # Merge phases
        if 'phases' in merge_sections and 'phases' in base_templates:
            phases_data = {}
            for phase_name, phase_content in base_templates['phases'].items():
                if 'phase_overrides' in overrides and phase_name in overrides['phase_overrides']:
                    phase_content = self.deep_merge(phase_content, overrides['phase_overrides'][phase_name])
                phases_data[phase_name] = phase_content
            merged['overrides']['phases'] = phases_data
            print(f"✓ Merged {len(phases_data)} phases")
       
        # Merge project-specific data (excluding project_info)
        if 'project_specific' in merge_sections and 'project_specific' in overrides:
            project_data = overrides['project_specific'].copy()
            # Remove project_info if it was already added separately
            if 'info' in merge_sections and 'project_info' in project_data:
                del project_data['project_info']
            merged['overrides']['project_specific'] = project_data
            print(f"✓ Merged project-specific data (repos, contacts, channels)")
       
        # Save merged result
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.projects_path / project_name / "merged_config.json"
       
        if self.save_json(merged, output_path):
            print(f"\n✓ Successfully merged for project '{project_name}'")
            print(f"✓ Output: {output_path}")
            print(f"✓ Sections: {', '.join(merge_sections)}")
       
        return merged
   
    def list_templates(self) -> Dict[str, List[str]]:
        """List all available templates"""
        templates = {
            'roles': [],
            'regions': [],
            'phases': []
        }
       
        # List roles
        role_path = self.templates_path / "role"
        if role_path.exists():
            templates['roles'] = [f.stem for f in role_path.glob("*.json")]
       
        # List regions
        region_path = self.templates_path / "region"
        if region_path.exists():
            templates['regions'] = [f.stem for f in region_path.glob("*.json")]
       
        # List phases
        phase_path = self.templates_path / "phase"
        if phase_path.exists():
            templates['phases'] = [f.stem for f in phase_path.glob("*.json")]
       
        return templates
   
    def list_projects(self) -> List[str]:
        """List all available projects"""
        if not self.projects_path.exists():
            return []
        return [d.name for d in self.projects_path.iterdir() if d.is_dir()]
 
 
def main():
    """Command-line interface for template merger"""
    import argparse
   
    parser = argparse.ArgumentParser(
        description='Merge onboarding templates with project-specific overrides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge AC1 project with ac1_template
  python merge_template.py --template ac1_template --project AC1
 
  # Merge with custom output
  python merge_template.py --template ac1_template --project AC1 --output custom_output.json
 
  # List available templates and projects
  python merge_template.py --list
        """
    )
   
    parser.add_argument('--template', '-t', help='Template name to use')
    parser.add_argument('--project', '-p', help='Project name to merge into')
    parser.add_argument('--output', '-o', help='Custom output file path')
    parser.add_argument('--list', '-l', action='store_true', help='List available templates and projects')
    parser.add_argument('--base-path', default='onboarding', help='Base path for onboarding directory')
   
    args = parser.parse_args()
   
    merger = TemplateMerger(base_path=args.base_path)
   
    # List mode
    if args.list:
        print("\n=== Available Templates ===")
        templates = merger.list_templates()
        print(f"\nRoles: {', '.join(templates['roles'])}")
        print(f"Regions: {', '.join(templates['regions'])}")
        print(f"Phases: {', '.join(templates['phases'])}")
       
        print("\n=== Available Projects ===")
        projects = merger.list_projects()
        print(f"Projects: {', '.join(projects)}")
        return 0
   
    # Merge mode
    if not args.template or not args.project:
        parser.error("Both --template and --project are required for merge operation")
        return 1
   
    try:
        result = merger.merge_with_overrides(
            template_name=args.template,
            project_name=args.project,
            output_file=args.output
        )
       
        if result:
            print("\n=== Merge Summary ===")
            if 'role' in result:
                print(f"Role: {result['role'].get('role', 'N/A')}")
            if 'region' in result:
                print(f"Region: {result['region'].get('region', 'N/A')}")
            if 'phases' in result:
                print(f"Phases: {', '.join(result['phases'].keys())}")
            return 0
        else:
            print("\n✗ Merge failed")
            return 1
           
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return 1
 
 
if __name__ == "__main__":
    sys.exit(main())
 
 