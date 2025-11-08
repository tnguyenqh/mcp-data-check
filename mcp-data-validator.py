from fastmcp import FastMCP
from pathlib import Path
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import re
from typing import List, Dict
from datetime import datetime
from collections import Counter

mcp = FastMCP(name="mcp-data-validator")

# Resource 1: Field descriptions from Data rules folder
@mcp.resource("dict://field-descriptions")
def get_field_descriptions() -> str:
    """Human-readable field descriptions and business context."""
    file_path = Path(__file__).parent / "Data rules" / "field_descriptions.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Resource 2: Validation rules from Data rules folder
@mcp.resource("schema://validation-rules")
def get_validation_rules() -> str:
    """Structured validation rules schema with constraints."""
    file_path = Path(__file__).parent / "Data rules" / "validation_rules.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Resource 3: Field mappings from Data rules folder
@mcp.resource("mapping://field-mappings")
def get_field_mappings() -> str:
    """Field name mappings to handle variations across different datasets."""
    file_path = Path(__file__).parent / "Data rules" / "field_mappings.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
# Tool 1: Read claims data file (CSV or Excel) and return as JSON
@mcp.tool()
def read_claims_file(
    file_path: str,
    sheet_name: Optional[str] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
    preview_rows: Optional[int] = None
) -> dict:
    """
    Reads CSV or Excel file containing insurance claims data and returns it as JSON format.
    
    Args:
        file_path: Path to the CSV or Excel file
        sheet_name: Sheet name for Excel files (default: first sheet)
        delimiter: Delimiter for CSV files (default: comma)
        encoding: File encoding (default: utf-8)
        preview_rows: Number of rows to preview (optional, returns all if not specified)
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating if file was read successfully
        - data: List of dictionaries representing rows
        - columns: List of column names
        - row_count: Total number of rows
        - file_type: Type of file read (csv or excel)
        - error: Error message if failed
    """
    try:
        file_path_obj = Path(file_path)
        
        # Check if file exists
        if not file_path_obj.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "data": [],
                "columns": [],
                "row_count": 0
            }
        
        # Determine file type and read accordingly
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
            file_type = "csv"
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
            file_type = "excel"
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_extension}. Only CSV and Excel files are supported.",
                "data": [],
                "columns": [],
                "row_count": 0
            }
        
        # Get total row count before preview
        total_rows = len(df)
        
        # Apply preview limit if specified
        if preview_rows is not None and preview_rows > 0:
            df = df.head(preview_rows)
        
        # Convert to JSON-serializable format
        columns = df.columns.tolist()
        data = df.to_dict(orient='records')
        
        return {
            "success": True,
            "data": data,
            "columns": columns,
            "row_count": total_rows,
            "preview_count": len(data),
            "file_type": file_type,
            "file_name": file_path_obj.name
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}",
            "data": [],
            "columns": [],
            "row_count": 0
        }

# Tool 2: Map non-standard field names to standard field names
@mcp.tool()
def map_fields(
    columns: List[str],
    apply_mapping: bool = False,
    data: Optional[List[Dict]] = None
) -> dict:
    """
    Maps non-standard field names to standard field names using predefined aliases.
    
    Args:
        columns: List of column names from the dataset to map
        apply_mapping: If True and data is provided, returns data with mapped column names
        data: Optional list of dictionaries (records) to apply mapping to
    
    Returns:
        Dictionary containing:
        - mapped_fields: Dictionary mapping original -> standard field names
        - unmapped_fields: List of fields that couldn't be mapped
        - standard_fields_found: List of standard field names found
        - mapping_summary: Summary statistics
        - mapped_data: Data with renamed columns (only if apply_mapping=True and data provided)
    """
    try:
        # Load field mappings
        mappings_path = Path(__file__).parent / "Data rules" / "field_mappings.json"
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings_config = json.load(f)
        
        # Build mapping dictionary
        mapping_lookup = {}
        for mapping in mappings_config['mappings']:
            standard_name = mapping['standard_field_name']
            case_sensitive = mapping.get('case_sensitive', False)
            
            # Add the standard name itself
            if case_sensitive:
                mapping_lookup[standard_name] = standard_name
            else:
                mapping_lookup[standard_name.lower()] = standard_name
            
            # Add all aliases
            for alias in mapping['aliases']:
                if case_sensitive:
                    mapping_lookup[alias] = standard_name
                else:
                    mapping_lookup[alias.lower()] = standard_name
        
        # Normalize column names (remove spaces, hyphens, underscores for matching)
        def normalize_field_name(field_name: str) -> str:
            return field_name.replace('_', '').replace('-', '').replace(' ', '').lower()
        
        # Create normalized lookup
        normalized_lookup = {}
        for key, value in mapping_lookup.items():
            normalized_key = normalize_field_name(key)
            normalized_lookup[normalized_key] = value
        
        # Map the columns
        mapped_fields = {}
        unmapped_fields = []
        
        for col in columns:
            col_lower = col.lower()
            col_normalized = normalize_field_name(col)
            
            # Try exact match first (case-insensitive)
            if col_lower in mapping_lookup:
                mapped_fields[col] = mapping_lookup[col_lower]
            # Try normalized match
            elif col_normalized in normalized_lookup:
                mapped_fields[col] = normalized_lookup[col_normalized]
            else:
                unmapped_fields.append(col)
        
        # Get unique standard fields found
        standard_fields_found = list(set(mapped_fields.values()))
        
        # Create mapping summary
        mapping_summary = {
            "total_columns": len(columns),
            "mapped_count": len(mapped_fields),
            "unmapped_count": len(unmapped_fields),
            "standard_fields_identified": len(standard_fields_found),
            "mapping_rate": f"{(len(mapped_fields) / len(columns) * 100):.1f}%" if columns else "0%"
        }
        
        result = {
            "success": True,
            "mapped_fields": mapped_fields,
            "unmapped_fields": unmapped_fields,
            "standard_fields_found": sorted(standard_fields_found),
            "mapping_summary": mapping_summary
        }
        
        # Apply mapping to data if requested
        if apply_mapping and data is not None:
            mapped_data = []
            for record in data:
                new_record = {}
                for old_col, value in record.items():
                    new_col = mapped_fields.get(old_col, old_col)
                    new_record[new_col] = value
                mapped_data.append(new_record)
            result["mapped_data"] = mapped_data
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error mapping fields: {str(e)}",
            "mapped_fields": {},
            "unmapped_fields": columns,
            "standard_fields_found": [],
            "mapping_summary": {}
        }

# Tool 3: Get validation rules for specified fields   
@mcp.tool()
def get_field_rules(
    field_names: Optional[Union[str, List[str]]] = None
) -> dict:
    """
    Returns validation rules, constraints, allowed values, and descriptions for insurance claim fields.
    
    Args:
        field_names: Single field name (string) or list of field names to get rules for.
                    If None or empty, returns rules for all fields.
    
    Returns:
        Dictionary containing:
        - fields: List of field rule objects with validation details
        - cross_field_validations: Rules that involve multiple fields
        - total_fields: Number of fields returned
        - schema_info: Schema version and metadata
    """
    try:
        # Load validation rules
        rules_path = Path(__file__).parent / "Data rules" / "validation_rules.json"
        with open(rules_path, 'r', encoding='utf-8') as f:
            validation_config = json.load(f)
        
        # Load field descriptions
        descriptions_path = Path(__file__).parent / "Data rules" / "field_descriptions.txt"
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            descriptions_text = f.read()
        
        # Convert field_names to list if it's a string
        if isinstance(field_names, str):
            field_names = [field_names]
        
        # Normalize field names for case-insensitive matching
        normalized_search = None
        if field_names:
            normalized_search = [name.upper().strip() for name in field_names]
        
        # Filter fields if specific field names provided
        fields_data = validation_config.get('fields', [])
        if normalized_search:
            filtered_fields = []
            for field in fields_data:
                if field['field_name'].upper() in normalized_search:
                    filtered_fields.append(field)
            fields_data = filtered_fields
        
        # Enhance fields with descriptions from text file
        for field in fields_data:
            field_name = field['field_name']
            # Extract description from field_descriptions.txt
            if f"[{field_name}]" in descriptions_text:
                start_idx = descriptions_text.index(f"[{field_name}]")
                # Find the next field marker or end
                next_field_idx = descriptions_text.find("\n[", start_idx + 1)
                if next_field_idx == -1:
                    next_field_idx = len(descriptions_text)
                
                field_section = descriptions_text[start_idx:next_field_idx].strip()
                field['description'] = field_section
        
        # Filter cross-field validations if specific fields requested
        cross_field_validations = validation_config.get('cross_field_validations', [])
        if normalized_search:
            filtered_cross_validations = []
            for validation in cross_field_validations:
                # Include cross-field validation if any of the requested fields are involved
                validation_fields = [f.upper() for f in validation.get('fields', [])]
                if any(search_field in validation_fields for search_field in normalized_search):
                    filtered_cross_validations.append(validation)
            cross_field_validations = filtered_cross_validations
        
        # Prepare result
        result = {
            "success": True,
            "fields": fields_data,
            "cross_field_validations": cross_field_validations,
            "total_fields": len(fields_data),
            "schema_info": {
                "schema_version": validation_config.get('schema_version', '1.0'),
                "last_updated": validation_config.get('last_updated'),
                "description": validation_config.get('description')
            }
        }
        
        # Add warning if no fields found for requested names
        if field_names and len(fields_data) == 0:
            result["warning"] = f"No fields found matching: {', '.join(field_names)}"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving field rules: {str(e)}",
            "fields": [],
            "cross_field_validations": [],
            "total_fields": 0
        }

# Tool 4: Validate claims data against all rules
@mcp.tool()
def validate_claims_data(
    data: List[Dict],
    auto_map_fields: bool = True,
    return_detailed_errors: bool = True
) -> dict:
    """
    Validates insurance claims data against all validation rules.
    Leverages existing tools: map_fields and get_field_rules.
    
    Args:
        data: List of dictionaries representing claim records
        auto_map_fields: Automatically map non-standard field names to standard names
        return_detailed_errors: Include row-by-row error details in response
    
    Returns:
        Dictionary containing:
        - success: Overall validation success status
        - validation_summary: Statistics about validation results
        - errors: List of validation errors with details
        - warnings: List of validation warnings
        - data_quality_metrics: Quality scores and metrics
    """
    try:
        if not data or len(data) == 0:
            return {
                "success": False,
                "error": "No data provided for validation"
            }
        
        # Use map_fields tool to standardize field names
        if auto_map_fields:
            columns = list(data[0].keys())
            mapping_result = map_fields(columns, apply_mapping=True, data=data)
            if mapping_result['success']:
                data = mapping_result.get('mapped_data', data)
                mapped_fields_info = {
                    "mapped_count": mapping_result['mapping_summary']['mapped_count'],
                    "unmapped_fields": mapping_result['unmapped_fields']
                }
            else:
                mapped_fields_info = {"error": "Field mapping failed"}
        else:
            mapped_fields_info = {"mapping": "disabled"}
        
        # Use get_field_rules tool to retrieve validation rules
        rules_result = get_field_rules()
        if not rules_result['success']:
            return {
                "success": False,
                "error": "Failed to load validation rules"
            }
        
        field_rules = {field['field_name']: field for field in rules_result['fields']}
        cross_validations = rules_result['cross_field_validations']
        
        # Initialize validation tracking
        errors = []
        warnings = []
        total_records = len(data)
        valid_records = 0
        
        # Track unique values for uniqueness constraints
        unique_trackers = {}
        for field_name, field_config in field_rules.items():
            if field_config.get('constraints', {}).get('unique', False):
                unique_trackers[field_name] = {}
        
        # Validate each record
        for row_idx, record in enumerate(data, start=1):
            row_errors = []
            row_warnings = []
            
            # Validate each field
            for field_name, field_config in field_rules.items():
                value = record.get(field_name)
                
                # Check required fields
                if field_config['required'] and (value is None or str(value).strip() == ''):
                    if field_config.get('nullable', False):
                        continue
                    row_errors.append({
                        "row": row_idx,
                        "field": field_name,
                        "error_type": "REQUIRED_FIELD_MISSING",
                        "message": f"Required field '{field_name}' is missing or empty"
                    })
                    continue
                
                # Skip validation if optional and null
                if value is None or str(value).strip() == '':
                    continue
                
                # Check uniqueness
                if field_name in unique_trackers:
                    value_key = str(value).strip()
                    if value_key in unique_trackers[field_name]:
                        row_errors.append({
                            "row": row_idx,
                            "field": field_name,
                            "value": value,
                            "error_type": "DUPLICATE_VALUE",
                            "message": f"Duplicate value found. Previously seen at row {unique_trackers[field_name][value_key]}"
                        })
                    else:
                        unique_trackers[field_name][value_key] = row_idx
                
                # Validate by data type
                data_type = field_config['data_type']
                constraints = field_config.get('constraints', {})
                
                if data_type == 'string':
                    value_str = str(value).strip()
                    min_len = constraints.get('min_length')
                    max_len = constraints.get('max_length')
                    pattern = constraints.get('pattern')
                    
                    if min_len and len(value_str) < min_len:
                        row_errors.append({
                            "row": row_idx,
                            "field": field_name,
                            "value": value_str,
                            "error_type": "LENGTH_TOO_SHORT",
                            "message": f"Value length {len(value_str)} is less than minimum {min_len}"
                        })
                    
                    if max_len and len(value_str) > max_len:
                        row_errors.append({
                            "row": row_idx,
                            "field": field_name,
                            "value": value_str,
                            "error_type": "LENGTH_TOO_LONG",
                            "message": f"Value length {len(value_str)} exceeds maximum {max_len}"
                        })
                    
                    if pattern and not re.match(pattern, value_str):
                        row_errors.append({
                            "row": row_idx,
                            "field": field_name,
                            "value": value_str,
                            "error_type": "PATTERN_MISMATCH",
                            "message": f"Value does not match required pattern: {pattern}"
                        })
                
                elif data_type == 'date':
                    date_formats = constraints.get('format', ['%Y-%m-%d', '%m/%d/%Y'])
                    parsed_date = None
                    
                    for fmt in date_formats:
                        try:
                            if fmt == 'YYYY-MM-DD':
                                fmt = '%Y-%m-%d'
                            elif fmt == 'MM/DD/YYYY':
                                fmt = '%m/%d/%Y'
                            parsed_date = datetime.strptime(str(value), fmt)
                            break
                        except:
                            continue
                    
                    if not parsed_date:
                        row_errors.append({
                            "row": row_idx,
                            "field": field_name,
                            "value": value,
                            "error_type": "INVALID_DATE_FORMAT",
                            "message": f"Invalid date format. Expected: {', '.join(date_formats)}"
                        })
                    else:
                        # Check date constraints
                        if not constraints.get('future_allowed', True) and parsed_date > datetime.now():
                            row_errors.append({
                                "row": row_idx,
                                "field": field_name,
                                "value": value,
                                "error_type": "FUTURE_DATE_NOT_ALLOWED",
                                "message": "Date cannot be in the future"
                            })
                        
                        min_date_str = constraints.get('min_date')
                        if min_date_str and min_date_str != 'current_date':
                            min_date = datetime.strptime(min_date_str, '%Y-%m-%d')
                            if parsed_date < min_date:
                                row_errors.append({
                                    "row": row_idx,
                                    "field": field_name,
                                    "value": value,
                                    "error_type": "DATE_TOO_EARLY",
                                    "message": f"Date is before minimum allowed date: {min_date_str}"
                                })
                
                elif data_type == 'numeric':
                    # Clean numeric value
                    cleaned_value = str(value).replace('$', '').replace(',', '').strip()
                    try:
                        numeric_value = float(cleaned_value)
                        
                        min_val = constraints.get('min_value')
                        max_val = constraints.get('max_value')
                        
                        if min_val is not None and numeric_value < min_val:
                            row_errors.append({
                                "row": row_idx,
                                "field": field_name,
                                "value": value,
                                "error_type": "VALUE_BELOW_MINIMUM",
                                "message": f"Value {numeric_value} is below minimum {min_val}"
                            })
                        
                        if max_val is not None and numeric_value > max_val:
                            row_errors.append({
                                "row": row_idx,
                                "field": field_name,
                                "value": value,
                                "error_type": "VALUE_ABOVE_MAXIMUM",
                                "message": f"Value {numeric_value} exceeds maximum {max_val}"
                            })
                    except:
                        row_errors.append({
                            "row": row_idx,
                            "field": field_name,
                            "value": value,
                            "error_type": "INVALID_NUMERIC_VALUE",
                            "message": "Value cannot be converted to number"
                        })
                
                elif data_type == 'categorical':
                    allowed_values = constraints.get('allowed_values', [])
                    case_sensitive = constraints.get('case_sensitive', False)
                    
                    if case_sensitive:
                        allowed_check = allowed_values
                        check_value = str(value)
                    else:
                        allowed_check = [v.upper() for v in allowed_values]
                        check_value = str(value).upper()
                    
                    if check_value not in allowed_check:
                        row_errors.append({
                            "row": row_idx,
                            "field": field_name,
                            "value": value,
                            "error_type": "INVALID_CATEGORY_VALUE",
                            "message": f"Value not in allowed list: {', '.join(allowed_values)}"
                        })
            
            # Apply cross-field validations from get_field_rules
            for cv in cross_validations:
                rule_id = cv['rule_id']
                
                # CV001: LOSS_DATE <= LODGED
                if rule_id == 'CV001':
                    loss_date = record.get('LOSS_DATE')
                    lodged = record.get('LODGED')
                    if loss_date and lodged:
                        try:
                            for fmt in ['%Y-%m-%d', '%m/%d/%Y']:
                                try:
                                    ld = datetime.strptime(str(loss_date), fmt)
                                    lg = datetime.strptime(str(lodged), fmt)
                                    if ld > lg:
                                        row_errors.append({
                                            "row": row_idx,
                                            "field": "LOSS_DATE, LODGED",
                                            "error_type": "CROSS_FIELD_VALIDATION",
                                            "message": cv['description']
                                        })
                                    break
                                except:
                                    continue
                        except:
                            pass
                
                # CV002: LODGED <= CLAIM_CLOSED_DATE
                elif rule_id == 'CV002':
                    lodged = record.get('LODGED')
                    closed = record.get('CLAIM_CLOSED_DATE')
                    if lodged and closed:
                        try:
                            for fmt in ['%Y-%m-%d', '%m/%d/%Y']:
                                try:
                                    lg = datetime.strptime(str(lodged), fmt)
                                    cl = datetime.strptime(str(closed), fmt)
                                    if lg > cl:
                                        row_errors.append({
                                            "row": row_idx,
                                            "field": "LODGED, CLAIM_CLOSED_DATE",
                                            "error_type": "CROSS_FIELD_VALIDATION",
                                            "message": cv['description']
                                        })
                                    break
                                except:
                                    continue
                        except:
                            pass
                
                # CV003: WAITING_PERIOD and BENEFIT_PERIOD required for IP/GSC
                elif rule_id == 'CV003':
                    benefit_type = str(record.get('BENEFIT_TYPE', '')).upper()
                    if benefit_type in ['IP/GSC', 'IP', 'GSC']:
                        if not record.get('WAITING_PERIOD'):
                            row_errors.append({
                                "row": row_idx,
                                "field": "WAITING_PERIOD",
                                "error_type": "CONDITIONAL_REQUIRED",
                                "message": "WAITING_PERIOD required for IP/GSC benefit type"
                            })
                        if not record.get('BENEFIT_PERIOD'):
                            row_errors.append({
                                "row": row_idx,
                                "field": "BENEFIT_PERIOD",
                                "error_type": "CONDITIONAL_REQUIRED",
                                "message": "BENEFIT_PERIOD required for IP/GSC benefit type"
                            })
                
                # CV004: Age calculation validation
                elif rule_id == 'CV004':
                    dob = record.get('DATE_OF_BIRTH')
                    loss_date = record.get('LOSS_DATE')
                    age_at_event = record.get('AGE_AT_EVENT')
                    
                    if dob and loss_date and age_at_event:
                        try:
                            for fmt in ['%Y-%m-%d', '%m/%d/%Y']:
                                try:
                                    dob_dt = datetime.strptime(str(dob), fmt)
                                    loss_dt = datetime.strptime(str(loss_date), fmt)
                                    calculated_age = (loss_dt - dob_dt).days // 365
                                    reported_age = int(age_at_event)
                                    
                                    if abs(calculated_age - reported_age) > 1:
                                        row_warnings.append({
                                            "row": row_idx,
                                            "field": "AGE_AT_EVENT",
                                            "warning_type": "AGE_MISMATCH",
                                            "message": f"AGE_AT_EVENT ({reported_age}) doesn't match calculated age ({calculated_age})"
                                        })
                                    break
                                except:
                                    continue
                        except:
                            pass
                
                # CV005: PAID_TO_DATE should not exceed SUM_INSURED for lump sum
                elif rule_id == 'CV005':
                    benefit_type = str(record.get('BENEFIT_TYPE', '')).upper()
                    if benefit_type in ['DEATH', 'TPD']:
                        try:
                            paid_to_date = float(str(record.get('PAID_TO_DATE', 0)).replace('$', '').replace(',', '').strip() or 0)
                            sum_insured = float(str(record.get('SUM_INSURED', 0)).replace('$', '').replace(',', '').strip() or 0)
                            if sum_insured > 0 and paid_to_date > sum_insured:
                                row_warnings.append({
                                    "row": row_idx,
                                    "field": "PAID_TO_DATE",
                                    "warning_type": "BUSINESS_LOGIC",
                                    "message": f"PAID_TO_DATE (${paid_to_date:,.2f}) exceeds SUM_INSURED (${sum_insured:,.2f})"
                                })
                        except:
                            pass
            
            # Track record validity
            if len(row_errors) == 0:
                valid_records += 1
            
            errors.extend(row_errors)
            warnings.extend(row_warnings)
        
        # Calculate validation summary
        validation_summary = {
            "total_records": total_records,
            "valid_records": valid_records,
            "invalid_records": total_records - valid_records,
            "total_errors": len(errors),
            "total_warnings": len(warnings),
            "validation_rate": f"{(valid_records / total_records * 100):.2f}%" if total_records > 0 else "0%"
        }
        
        # Data quality metrics
        data_quality_metrics = {
            "completeness": f"{(valid_records / total_records * 100):.2f}%" if total_records > 0 else "0%",
            "error_density": f"{(len(errors) / total_records):.2f}" if total_records > 0 else "0",
            "fields_validated": len(field_rules)
        }
        
        result = {
            "success": len(errors) == 0,
            "validation_summary": validation_summary,
            "data_quality_metrics": data_quality_metrics,
            "field_mapping_info": mapped_fields_info,
            "warnings": warnings
        }
        
        if return_detailed_errors:
            result["errors"] = errors
        else:
            # Return only summary of error types
            error_types = {}
            for err in errors:
                err_type = err['error_type']
                error_types[err_type] = error_types.get(err_type, 0) + 1
            result["error_summary"] = error_types
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Validation failed: {str(e)}",
            "validation_summary": {},
            "errors": [],
            "warnings": []
        }

# Tool 5: Analyze data quality beyond validation
@mcp.tool()
def check_data_quality(
    data: List[Dict],
    auto_map_fields: bool = True
) -> dict:
    """
    Analyzes dataset for data quality issues beyond validation rules.
    Provides insights on completeness, distributions, outliers, and inconsistencies.
    
    Args:
        data: List of dictionaries representing claim records
        auto_map_fields: Automatically map non-standard field names to standard names
    
    Returns:
        Dictionary containing:
        - quality_score: Overall data quality score (0-100)
        - completeness_analysis: Field-by-field completeness rates
        - distribution_analysis: Value distributions for categorical fields
        - outlier_detection: Potential outliers in numeric fields
        - consistency_issues: Data consistency problems
        - recommendations: Suggested improvements
    """
    try:
        if not data or len(data) == 0:
            return {
                "success": False,
                "error": "No data provided for quality analysis"
            }
        
        # Use map_fields tool to standardize field names
        if auto_map_fields:
            columns = list(data[0].keys())
            mapping_result = map_fields(columns, apply_mapping=True, data=data)
            if mapping_result['success']:
                data = mapping_result.get('mapped_data', data)
        
        # Use get_field_rules to understand field types
        rules_result = get_field_rules()
        if not rules_result['success']:
            return {
                "success": False,
                "error": "Failed to load field rules for quality analysis"
            }
        
        field_rules = {field['field_name']: field for field in rules_result['fields']}
        total_records = len(data)
        
        # 1. COMPLETENESS ANALYSIS
        completeness_analysis = {}
        for field_name, field_config in field_rules.items():
            non_null_count = 0
            for record in data:
                value = record.get(field_name)
                if value is not None and str(value).strip() != '':
                    non_null_count += 1
            
            completeness_rate = (non_null_count / total_records) * 100
            completeness_analysis[field_name] = {
                "completeness_rate": f"{completeness_rate:.2f}%",
                "non_null_count": non_null_count,
                "null_count": total_records - non_null_count,
                "is_required": field_config['required']
            }
        
        # 2. DISTRIBUTION ANALYSIS (for categorical fields)
        distribution_analysis = {}
        for field_name, field_config in field_rules.items():
            if field_config['data_type'] == 'categorical':
                values = []
                for record in data:
                    value = record.get(field_name)
                    if value is not None and str(value).strip() != '':
                        values.append(str(value).upper())
                
                if values:
                    value_counts = Counter(values)
                    total_valid = len(values)
                    
                    distribution_analysis[field_name] = {
                        "unique_values": len(value_counts),
                        "most_common": [
                            {"value": val, "count": count, "percentage": f"{(count/total_valid)*100:.2f}%"}
                            for val, count in value_counts.most_common(5)
                        ],
                        "expected_values": field_config.get('constraints', {}).get('allowed_values', [])
                    }
        
        # 3. OUTLIER DETECTION (for numeric fields)
        outlier_detection = {}
        for field_name, field_config in field_rules.items():
            if field_config['data_type'] == 'numeric':
                values = []
                for record in data:
                    value = record.get(field_name)
                    if value is not None and str(value).strip() != '':
                        try:
                            cleaned_value = str(value).replace('$', '').replace(',', '').strip()
                            numeric_value = float(cleaned_value)
                            values.append(numeric_value)
                        except:
                            pass
                
                if len(values) > 0:
                    values.sort()
                    min_val = values[0]
                    max_val = values[-1]
                    mean_val = sum(values) / len(values)
                    
                    # Calculate median
                    n = len(values)
                    median_val = values[n//2] if n % 2 == 1 else (values[n//2-1] + values[n//2]) / 2
                    
                    # Simple outlier detection using IQR
                    q1_idx = n // 4
                    q3_idx = 3 * n // 4
                    q1 = values[q1_idx]
                    q3 = values[q3_idx]
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = [v for v in values if v < lower_bound or v > upper_bound]
                    
                    outlier_detection[field_name] = {
                        "min": min_val,
                        "max": max_val,
                        "mean": f"{mean_val:.2f}",
                        "median": f"{median_val:.2f}",
                        "outlier_count": len(outliers),
                        "outlier_percentage": f"{(len(outliers)/len(values))*100:.2f}%",
                        "has_outliers": len(outliers) > 0
                    }
        
        # 4. CONSISTENCY ISSUES
        consistency_issues = []
        
        # Check for duplicate claim numbers
        claim_numbers = []
        for record in data:
            claim_num = record.get('CLAIM_NUMBER')
            if claim_num:
                claim_numbers.append(str(claim_num))
        
        duplicate_claims = [item for item, count in Counter(claim_numbers).items() if count > 1]
        if duplicate_claims:
            consistency_issues.append({
                "issue": "Duplicate Claim Numbers",
                "severity": "HIGH",
                "count": len(duplicate_claims),
                "description": f"Found {len(duplicate_claims)} claim numbers that appear multiple times"
            })
        
        # Check for date inconsistencies
        date_inconsistencies = 0
        for record in data:
            dob = record.get('DATE_OF_BIRTH')
            loss_date = record.get('LOSS_DATE')
            
            if dob and loss_date:
                try:
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            dob_dt = datetime.strptime(str(dob), fmt)
                            loss_dt = datetime.strptime(str(loss_date), fmt)
                            # Check if loss date is before birth date
                            if loss_dt < dob_dt:
                                date_inconsistencies += 1
                            break
                        except:
                            continue
                except:
                    pass
        
        if date_inconsistencies > 0:
            consistency_issues.append({
                "issue": "Date Logic Errors",
                "severity": "HIGH",
                "count": date_inconsistencies,
                "description": f"Found {date_inconsistencies} records where LOSS_DATE is before DATE_OF_BIRTH"
            })
        
        # Check for unrealistic ages
        unrealistic_ages = 0
        for record in data:
            age = record.get('AGE_AT_EVENT')
            if age:
                try:
                    age_val = int(age)
                    if age_val < 0 or age_val > 120:
                        unrealistic_ages += 1
                except:
                    pass
        
        if unrealistic_ages > 0:
            consistency_issues.append({
                "issue": "Unrealistic Ages",
                "severity": "MEDIUM",
                "count": unrealistic_ages,
                "description": f"Found {unrealistic_ages} records with AGE_AT_EVENT outside 0-120 range"
            })
        
        # 5. CALCULATE OVERALL QUALITY SCORE
        quality_factors = []
        
        # Completeness score (average completeness of required fields)
        required_completeness = [
            float(info['completeness_rate'].rstrip('%')) 
            for field, info in completeness_analysis.items() 
            if info['is_required']
        ]
        completeness_score = sum(required_completeness) / len(required_completeness) if required_completeness else 0
        quality_factors.append(completeness_score)
        
        # Consistency score (penalize for issues)
        consistency_score = 100
        for issue in consistency_issues:
            if issue['severity'] == 'HIGH':
                consistency_score -= min(30, issue['count'] / total_records * 100)
            elif issue['severity'] == 'MEDIUM':
                consistency_score -= min(15, issue['count'] / total_records * 100)
        consistency_score = max(0, consistency_score)
        quality_factors.append(consistency_score)
        
        # Overall quality score
        overall_quality_score = sum(quality_factors) / len(quality_factors)
        
        # 6. RECOMMENDATIONS
        recommendations = []
        
        # Completeness recommendations
        low_completeness_fields = [
            field for field, info in completeness_analysis.items()
            if float(info['completeness_rate'].rstrip('%')) < 80 and info['is_required']
        ]
        if low_completeness_fields:
            recommendations.append({
                "priority": "HIGH",
                "category": "Completeness",
                "recommendation": f"Improve data collection for fields: {', '.join(low_completeness_fields)}",
                "impact": "Required fields should have >95% completeness"
            })
        
        # Outlier recommendations
        high_outlier_fields = [
            field for field, info in outlier_detection.items()
            if float(info['outlier_percentage'].rstrip('%')) > 5
        ]
        if high_outlier_fields:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Data Quality",
                "recommendation": f"Review outliers in: {', '.join(high_outlier_fields)}",
                "impact": "Outliers may indicate data entry errors or exceptional cases"
            })
        
        # Consistency recommendations
        if consistency_issues:
            recommendations.append({
                "priority": "HIGH",
                "category": "Data Consistency",
                "recommendation": "Address data consistency issues found in analysis",
                "impact": "Inconsistent data can lead to incorrect analysis and reporting"
            })
        
        # If quality is good
        if overall_quality_score >= 90 and not recommendations:
            recommendations.append({
                "priority": "INFO",
                "category": "Overall",
                "recommendation": "Data quality is excellent. Continue current data collection practices.",
                "impact": "Maintain high standards"
            })
        
        return {
            "success": True,
            "quality_score": f"{overall_quality_score:.2f}",
            "quality_grade": "Excellent" if overall_quality_score >= 90 else "Good" if overall_quality_score >= 75 else "Fair" if overall_quality_score >= 60 else "Poor",
            "total_records_analyzed": total_records,
            "completeness_analysis": completeness_analysis,
            "distribution_analysis": distribution_analysis,
            "outlier_detection": outlier_detection,
            "consistency_issues": consistency_issues,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Data quality check failed: {str(e)}"
        }

# Prompt 1: Guide for comprehensive claims data validation workflow
@mcp.prompt()
def validate_claims_workflow_prompt(file_path: str) -> str:
    """
    Comprehensive workflow: Map fields → Get rules → Validate → Quality check → Summary
    
    Args:
        file_path: Path to the CSV or Excel file to validate
    """
    return f"""You are validating insurance claims data following a specific workflow.

File to validate: {file_path}

STRICT WORKFLOW - Follow these steps in exact order:

STEP 1: READ THE FILE
- Tool: read_claims_file(file_path="{file_path}")
- Confirm file loaded successfully
- Note the total record count
- Extract column names from the data

STEP 2: MAP FIELD NAMES
- Tool: map_fields(columns=<columns from step 1>, apply_mapping=True, data=<data from step 1>)
- This standardizes all field names to match validation schema
- Report:
  * How many fields were mapped successfully
  * List any unmapped fields
  * Show the mapping (original → standard name)
- Use the mapped_data for all subsequent steps

STEP 3: GET FIELD DESCRIPTIONS AND VALIDATION RULES
- Tool: get_field_rules() [no parameters = get all fields]
- This loads all validation rules and field descriptions
- Report:
  * Total number of fields with validation rules
  * List all required fields
  * Mention key constraints (e.g., unique fields, date ranges)
  * Reference the resource "field-descriptions" for business context

STEP 4: PERFORM VALIDATION
- Tool: validate_claims_data(data=<mapped_data from step 2>, auto_map_fields=False, return_detailed_errors=True)
- Note: Set auto_map_fields=False since we already mapped in step 2
- Collect all validation results:
  * Validation summary statistics
  * All errors by type
  * All warnings
  * Field mapping information

STEP 5: RUN QUALITY CHECK
- Tool: check_data_quality(data=<mapped_data from step 2>, auto_map_fields=False)
- Note: Set auto_map_fields=False since we already mapped in step 2
- Collect quality analysis:
  * Overall quality score and grade
  * Completeness analysis per field
  * Distribution analysis for categorical fields
  * Outlier detection for numeric fields
  * Consistency issues found
  * Quality recommendations

STEP 6: OUTPUT VALIDATION SUMMARY
Present a comprehensive report with these sections:

## 1. FILE INFORMATION
- File name: {file_path}
- Total records: <count>
- Date processed: <current date>

## 2. FIELD MAPPING RESULTS
- Mapped fields: <count> / <total columns>
- Unmapped fields: <list if any>
- Mapping details: <show key mappings>

## 3. VALIDATION RULES APPLIED
- Total validation rules: <count>
- Required fields checked: <list>
- Cross-field validations: <count>

## 4. VALIDATION RESULTS
- Valid records: <count> (<percentage>%)
- Invalid records: <count> (<percentage>%)
- Total errors: <count>
- Total warnings: <count>

### Error Breakdown by Type:
<table or list of error types and counts>

### Top 10 Errors (with row numbers):
<detailed errors>

## 5. DATA QUALITY ASSESSMENT
- Overall Quality Score: <score> / 100
- Quality Grade: <Excellent/Good/Fair/Poor>

### Completeness Analysis:
- Fields below 100% completeness
- Critical missing data issues

### Data Consistency:
- Duplicate claim numbers: <count if any>
- Date logic errors: <count if any>
- Other consistency issues: <list>

### Outlier Analysis:
- Fields with outliers: <list>
- Most significant outliers: <details>

## 6. RECOMMENDATIONS
Priority-ordered list of actions:
1. <HIGH priority items>
2. <MEDIUM priority items>
3. <LOW priority items>

## 7. OVERALL ASSESSMENT
- Status: [READY FOR USE / MINOR FIXES NEEDED / MAJOR CLEANUP REQUIRED]
- Next Steps: <specific actionable items>

---

IMPORTANT REMINDERS:
- Execute ALL steps in sequence - do not skip any step
- Use the mapped data from step 2 for steps 4 and 5
- Set auto_map_fields=False in steps 4 and 5 since mapping is already done
- Provide the complete summary output at the end
- Be thorough but organized with clear sections
- Use tables and bullet points for readability"""
