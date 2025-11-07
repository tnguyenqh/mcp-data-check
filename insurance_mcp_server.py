"""
Life Insurance Claims Data Validation MCP Server (FastMCP)

This MCP server provides tools for validating life insurance claim data,
with intelligent field recognition and comprehensive validation checks.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
from mcp.server.fastmcp import FastMCP

# Field name mappings for intelligent recognition
FIELD_MAPPINGS = {
    'claim_number': ['claim_number', 'claim_no', 'claimno', 'claim_id', 'claimid', 'policy_claim_no'],
    'policy_number': ['policy_number', 'policy_no', 'policyno', 'policy_id', 'policyid'],
    'claimant_name': ['claimant_name', 'name', 'claimant', 'insured_name', 'member_name'],
    'date_of_birth': ['date_of_birth', 'dob', 'birth_date', 'birthdate', 'date_birth'],
    'gender': ['gender', 'sex', 'claimant_gender', 'claimant_sex'],
    'event_date': ['event_date', 'loss_date', 'date_of_loss', 'incident_date', 'death_date', 'date_of_death'],
    'notification_date': ['notification_date', 'notified_date', 'report_date', 'reported_date', 'date_reported'],
    'paid_date': ['paid_date', 'payment_date', 'date_paid', 'settlement_date'],
    'claim_amount': ['claim_amount', 'amount', 'paid_amount', 'settlement_amount', 'benefit_amount'],
    'claim_status': ['claim_status', 'status', 'claim_state'],
    'age_at_event': ['age_at_event', 'age_at_loss', 'age_at_death', 'age'],
}

# Initialize FastMCP server
mcp = FastMCP("Insurance Claims Validator")

# Store loaded data globally
loaded_data: Optional[pd.DataFrame] = None
field_map: Dict[str, str] = {}


def normalize_field_name(field: str) -> str:
    """Normalize field names for comparison"""
    return field.lower().strip().replace(' ', '_').replace('-', '_')


def identify_fields(columns: List[str]) -> Dict[str, str]:
    """
    Intelligently map actual column names to standard field names
    Returns dict mapping standard_name -> actual_column_name
    """
    mapping = {}
    normalized_cols = {normalize_field_name(col): col for col in columns}
    
    for standard_name, variations in FIELD_MAPPINGS.items():
        for variation in variations:
            norm_variation = normalize_field_name(variation)
            if norm_variation in normalized_cols:
                mapping[standard_name] = normalized_cols[norm_variation]
                break
    
    return mapping


def parse_date(date_val: Any) -> Optional[datetime]:
    """Parse various date formats"""
    if pd.isna(date_val):
        return None
    
    if isinstance(date_val, datetime):
        return date_val
    
    if isinstance(date_val, str):
        # Try common formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_val, fmt)
            except ValueError:
                continue
    
    return None


def calculate_age(dob: datetime, reference_date: datetime) -> Optional[int]:
    """Calculate age at a reference date"""
    if not dob or not reference_date:
        return None
    
    age = reference_date.year - dob.year
    if (reference_date.month, reference_date.day) < (dob.month, dob.day):
        age -= 1
    return age


def validate_claim_data(df: pd.DataFrame, field_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Perform comprehensive validation checks on claim data
    """
    issues = {
        'critical': [],
        'warnings': [],
        'info': [],
        'summary': {}
    }
    
    total_records = len(df)
    issues['summary']['total_records'] = total_records
    
    # 1. Check for missing critical fields
    critical_fields = ['claim_number', 'event_date', 'claim_status']
    for field in critical_fields:
        if field in field_map:
            col = field_map[field]
            missing = df[col].isna().sum()
            if missing > 0:
                issues['critical'].append({
                    'check': 'Missing Critical Field',
                    'field': col,
                    'count': int(missing),
                    'percentage': round(missing / total_records * 100, 2)
                })
    
    # 2. Duplicate claim numbers
    if 'claim_number' in field_map:
        col = field_map['claim_number']
        duplicates = df[col].duplicated().sum()
        if duplicates > 0:
            dup_claims = df[df[col].duplicated(keep=False)][col].tolist()
            issues['critical'].append({
                'check': 'Duplicate Claim Numbers',
                'count': int(duplicates),
                'examples': dup_claims[:5]
            })
    
    # 3. Date validations
    date_fields = ['date_of_birth', 'event_date', 'notification_date', 'paid_date']
    parsed_dates = {}
    
    for field in date_fields:
        if field in field_map:
            col = field_map[field]
            df[f'{field}_parsed'] = df[col].apply(parse_date)
            parsed_dates[field] = f'{field}_parsed'
            
            # Check for invalid dates
            invalid = df[col].notna() & df[f'{field}_parsed'].isna()
            if invalid.sum() > 0:
                issues['warnings'].append({
                    'check': 'Invalid Date Format',
                    'field': col,
                    'count': int(invalid.sum()),
                    'examples': df[invalid][col].head(3).tolist()
                })
    
    # 4. Date sequence validations
    if 'event_date' in parsed_dates and 'notification_date' in parsed_dates:
        event_col = parsed_dates['event_date']
        notif_col = parsed_dates['notification_date']
        
        invalid_seq = (df[event_col].notna() & df[notif_col].notna() & 
                       (df[notif_col] < df[event_col]))
        
        if invalid_seq.sum() > 0:
            issues['critical'].append({
                'check': 'Notification Before Event',
                'description': 'Notification date is before event date',
                'count': int(invalid_seq.sum()),
                'claim_examples': df[invalid_seq][field_map.get('claim_number', df.columns[0])].head(3).tolist() if 'claim_number' in field_map else []
            })
    
    # 5. Event date should not be in future
    if 'event_date' in parsed_dates:
        event_col = parsed_dates['event_date']
        future_dates = df[event_col] > datetime.now()
        
        if future_dates.sum() > 0:
            issues['critical'].append({
                'check': 'Future Event Dates',
                'count': int(future_dates.sum()),
                'claim_examples': df[future_dates][field_map.get('claim_number', df.columns[0])].head(3).tolist() if 'claim_number' in field_map else []
            })
    
    # 6. Age validations
    if 'date_of_birth' in parsed_dates:
        dob_col = parsed_dates['date_of_birth']
        
        # Age too young (< 18 for life insurance)
        if 'event_date' in parsed_dates:
            event_col = parsed_dates['event_date']
            df['calculated_age'] = df.apply(
                lambda row: calculate_age(row[dob_col], row[event_col]) 
                if pd.notna(row[dob_col]) and pd.notna(row[event_col]) else None, 
                axis=1
            )
            
            too_young = (df['calculated_age'].notna()) & (df['calculated_age'] < 18)
            if too_young.sum() > 0:
                issues['warnings'].append({
                    'check': 'Age Below 18 at Event',
                    'count': int(too_young.sum()),
                    'claim_examples': df[too_young][field_map.get('claim_number', df.columns[0])].head(3).tolist() if 'claim_number' in field_map else []
                })
            
            # Age too old (> 120)
            too_old = (df['calculated_age'].notna()) & (df['calculated_age'] > 120)
            if too_old.sum() > 0:
                issues['critical'].append({
                    'check': 'Age Above 120 at Event',
                    'count': int(too_old.sum()),
                    'claim_examples': df[too_old][field_map.get('claim_number', df.columns[0])].head(3).tolist() if 'claim_number' in field_map else []
                })
            
            # Check if provided age matches calculated age
            if 'age_at_event' in field_map:
                age_col = field_map['age_at_event']
                age_mismatch = (df['calculated_age'].notna() & 
                               df[age_col].notna() & 
                               (abs(df['calculated_age'] - df[age_col]) > 1))
                
                if age_mismatch.sum() > 0:
                    issues['warnings'].append({
                        'check': 'Age Mismatch',
                        'description': 'Provided age does not match calculated age from DOB',
                        'count': int(age_mismatch.sum())
                    })
    
    # 7. Gender validation
    if 'gender' in field_map:
        col = field_map['gender']
        valid_genders = ['M', 'F', 'Male', 'Female', 'MALE', 'FEMALE', 'male', 'female']
        invalid_gender = ~df[col].isin(valid_genders) & df[col].notna()
        
        if invalid_gender.sum() > 0:
            issues['warnings'].append({
                'check': 'Invalid Gender Values',
                'count': int(invalid_gender.sum()),
                'invalid_values': df[invalid_gender][col].unique().tolist()
            })
    
    # 8. Claim amount validations
    if 'claim_amount' in field_map:
        col = field_map['claim_amount']
        
        # Negative amounts
        negative = (df[col].notna()) & (df[col] < 0)
        if negative.sum() > 0:
            issues['critical'].append({
                'check': 'Negative Claim Amounts',
                'count': int(negative.sum())
            })
        
        # Zero amounts for paid claims
        if 'claim_status' in field_map:
            status_col = field_map['claim_status']
            paid_zero = (df[status_col].str.lower().str.contains('paid|settled', na=False) & 
                        (df[col] == 0))
            
            if paid_zero.sum() > 0:
                issues['warnings'].append({
                    'check': 'Zero Amount for Paid Claims',
                    'count': int(paid_zero.sum())
                })
        
        # Outlier detection (amounts > 99.9th percentile)
        if df[col].notna().sum() > 0:
            threshold = df[col].quantile(0.999)
            outliers = (df[col] > threshold) & df[col].notna()
            
            if outliers.sum() > 0:
                issues['info'].append({
                    'check': 'Potential Outlier Amounts',
                    'description': f'Amounts exceeding {threshold:,.2f}',
                    'count': int(outliers.sum()),
                    'max_amount': float(df[col].max())
                })
    
    # 9. Notification lag analysis
    if 'event_date' in parsed_dates and 'notification_date' in parsed_dates:
        event_col = parsed_dates['event_date']
        notif_col = parsed_dates['notification_date']
        
        df['notification_lag'] = (df[notif_col] - df[event_col]).dt.days
        
        long_lag = (df['notification_lag'].notna()) & (df['notification_lag'] > 90)
        if long_lag.sum() > 0:
            issues['info'].append({
                'check': 'Long Notification Lag',
                'description': 'More than 90 days between event and notification',
                'count': int(long_lag.sum()),
                'avg_lag': float(df[long_lag]['notification_lag'].mean())
            })
    
    # 10. Status consistency checks
    if 'claim_status' in field_map:
        status_col = field_map['claim_status']
        
        # Paid claims without paid date
        if 'paid_date' in field_map:
            paid_col = field_map['paid_date']
            paid_parsed = parsed_dates.get('paid_date')
            
            paid_no_date = (df[status_col].str.lower().str.contains('paid|settled', na=False) & 
                           df[paid_parsed].isna())
            
            if paid_no_date.sum() > 0:
                issues['warnings'].append({
                    'check': 'Paid Status Without Paid Date',
                    'count': int(paid_no_date.sum())
                })
    
    # Summary statistics
    issues['summary']['critical_issues'] = len(issues['critical'])
    issues['summary']['warnings'] = len(issues['warnings'])
    issues['summary']['info_items'] = len(issues['info'])
    issues['summary']['fields_identified'] = len(field_map)
    
    return issues


# ==================== TOOLS ====================

@mcp.tool()
def load_claim_data(file_path: str) -> Dict[str, Any]:
    """
    Load insurance claim data from CSV file. Automatically identifies field names.
    
    Args:
        file_path: Path to the CSV file containing claim data
    
    Returns:
        Dictionary with loading status and field identification results
    """
    global loaded_data, field_map
    
    try:
        df = pd.read_csv(file_path)
        loaded_data = df
        field_map = identify_fields(df.columns.tolist())
        
        return {
            "success": True,
            "message": f"Loaded {len(df)} records",
            "total_records": len(df),
            "columns_found": df.columns.tolist(),
            "fields_identified": field_map,
            "unidentified_columns": [col for col in df.columns if col not in field_map.values()]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def validate_claim_data(include_examples: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive validation checks on loaded claim data.
    
    Args:
        include_examples: Include specific examples of issues in results (default: True)
    
    Returns:
        Dictionary containing critical issues, warnings, info items, and summary
    """
    if loaded_data is None:
        return {"error": "No data loaded. Use load_claim_data first."}
    
    validation_results = validate_claim_data(loaded_data, field_map)
    
    if not include_examples:
        # Remove examples from results
        for category in ['critical', 'warnings', 'info']:
            for issue in validation_results[category]:
                issue.pop('examples', None)
                issue.pop('claim_examples', None)
                issue.pop('invalid_values', None)
    
    return validation_results


@mcp.tool()
def get_claim_details(claim_number: str) -> Dict[str, Any]:
    """
    Retrieve details for a specific claim number.
    
    Args:
        claim_number: The claim number to look up
    
    Returns:
        Dictionary containing all available information for the claim
    """
    if loaded_data is None:
        return {"error": "No data loaded"}
    
    if 'claim_number' not in field_map:
        return {"error": "Claim number field not identified in data"}
    
    claim_col = field_map['claim_number']
    claim_data = loaded_data[loaded_data[claim_col] == claim_number]
    
    if len(claim_data) == 0:
        return {"error": f"Claim {claim_number} not found"}
    
    result = claim_data.iloc[0].to_dict()
    
    # Add calculated fields if possible
    if 'date_of_birth' in field_map and 'event_date' in field_map:
        dob = parse_date(result.get(field_map['date_of_birth']))
        event = parse_date(result.get(field_map['event_date']))
        if dob and event:
            result['calculated_age_at_event'] = calculate_age(dob, event)
    
    # Convert datetime objects to strings for JSON serialization
    for key, value in result.items():
        if isinstance(value, datetime):
            result[key] = value.strftime('%Y-%m-%d')
    
    return result


@mcp.tool()
def get_validation_summary() -> Dict[str, Any]:
    """
    Get a high-level summary of validation results.
    
    Returns:
        Dictionary with counts and lists of issue types
    """
    if loaded_data is None:
        return {"error": "No data loaded"}
    
    validation_results = validate_claim_data(loaded_data, field_map)
    
    return {
        "total_records": validation_results['summary']['total_records'],
        "fields_identified": validation_results['summary']['fields_identified'],
        "critical_issues_count": validation_results['summary']['critical_issues'],
        "warnings_count": validation_results['summary']['warnings'],
        "info_items_count": validation_results['summary']['info_items'],
        "critical_issues": [issue['check'] for issue in validation_results['critical']],
        "warnings": [issue['check'] for issue in validation_results['warnings']],
    }


# ==================== RESOURCES ====================

@mcp.resource("validation://field-mappings")
def get_field_mappings() -> str:
    """Shows how column names are mapped to standard fields"""
    return {
        "standard_fields": FIELD_MAPPINGS,
        "current_mapping": field_map if field_map else "No data loaded"
    }


@mcp.resource("validation://loaded-data-summary")
def get_loaded_data_summary() -> str:
    """Summary of currently loaded claim data"""
    if loaded_data is None:
        return {"error": "No data currently loaded"}
    
    return {
        "total_records": len(loaded_data),
        "columns": loaded_data.columns.tolist(),
        "identified_fields": field_map,
        "sample_data": loaded_data.head(3).to_dict(orient='records')
    }


# ==================== PROMPTS ====================

@mcp.prompt()
def validate_claims(focus_area: str = "all") -> List[Dict[str, str]]:
    """
    Validate insurance claim data and generate a comprehensive report.
    
    Args:
        focus_area: Specific area to focus on: dates, amounts, demographics, or all
    """
    prompt_text = f"""Please validate the loaded insurance claim data with focus on: {focus_area}

Generate a comprehensive validation report that includes:

1. **Data Quality Overview**
   - Total records analyzed
   - Field identification success rate
   - Overall data completeness

2. **Critical Issues** (must be fixed)
   - Missing required fields
   - Invalid date sequences
   - Duplicate claim numbers
   - Future event dates
   - Negative claim amounts

3. **Warnings** (should be reviewed)
   - Invalid data formats
   - Age discrepancies
   - Status inconsistencies
   - Missing paid dates for settled claims

4. **Informational Items**
   - Outlier amounts
   - Long notification lags
   - Statistical summaries

5. **Recommendations**
   - Priority actions
   - Data quality improvements
   - Process recommendations

Use the validate_claim_data tool to get the validation results, then format them into a clear, actionable report."""

    return [{"role": "user", "content": prompt_text}]


@mcp.prompt()
def investigate_claim(claim_number: str) -> List[Dict[str, str]]:
    """
    Deep dive into a specific claim number.
    
    Args:
        claim_number: The claim number to investigate
    """
    prompt_text = f"""Please investigate claim number: {claim_number}

Provide a detailed analysis including:

1. **Claim Details**
   - All available information for this claim
   - Key dates and amounts
   - Current status

2. **Data Quality Checks**
   - Any validation issues specific to this claim
   - Date sequence verification
   - Amount reasonability
   - Age calculations

3. **Red Flags**
   - Any anomalies or concerns
   - Comparison to similar claims
   - Missing information

4. **Timeline Analysis**
   - Event to notification lag
   - Notification to payment lag
   - Total claim lifecycle

Use the get_claim_details tool first, then analyze the results."""

    return [{"role": "user", "content": prompt_text}]


if __name__ == "__main__":
    mcp.run()