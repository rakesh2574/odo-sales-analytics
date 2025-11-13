"""
Sales Analytics Platform - Enhanced with Complete Query Examples
Handles Odoo table name mapping to actual database tables
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from openai import OpenAI
import sqlite3
import os
import re
from typing import Dict, List, Optional, Tuple, Set

# Page config
st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide"
)

# Minimal CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    h1 {color: #2c3e50; text-align: center; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# Table name mapping - No mapping needed, all tables use Odoo names
TABLE_NAME_MAPPING = {
    'res_partner': 'res_partner',
    'crm_team': 'crm_team',
    'res_users': 'res_users',
    'product_product': 'product_product',
    'product_template': 'product_template',
    'sale_order': 'sale_order',
    'sale_order_line': 'sale_order_line'
}


@st.cache_resource
def get_db_connection():
    """Get database connection"""
    db_file = 'sales_order.db'
    if not os.path.exists(db_file):
        st.error("Database not found. Please ensure sales_order.db is in the same folder.")
        return None
    return sqlite3.connect(db_file, check_same_thread=False)


@st.cache_data
def load_schema():
    """Load database schema"""
    try:
        with open('database_schema.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading schema: {e}")
        return None


def map_table_names_in_sql(sql: str) -> str:
    """
    Convert Odoo table names to actual database table names
    Preserves aliases and handles case-insensitive matching
    """
    mapped_sql = sql

    # Sort by length (descending) to avoid partial replacements
    for odoo_name, db_name in sorted(TABLE_NAME_MAPPING.items(), key=lambda x: len(x[0]), reverse=True):
        # Match table names with word boundaries, case-insensitive
        # Handles: FROM table, JOIN table, table AS alias, table alias
        pattern = r'\b' + re.escape(odoo_name) + r'\b'
        mapped_sql = re.sub(pattern, db_name, mapped_sql, flags=re.IGNORECASE)

    return mapped_sql


def get_valid_columns_and_tables(schema: dict) -> Tuple[Set[str], Set[str], Dict[str, List[str]], Dict[str, List[Dict]]]:
    """
    Extract all valid column names, table names, table-column mappings, and relationships
    Returns: (valid_columns, valid_tables, table_columns, relationships)
    """
    valid_tables = set()
    valid_columns = set()
    table_columns = {}
    relationships = {}

    for table in schema['tables']:
        # Add both Odoo table name and actual table name
        odoo_name = table['table_name']
        actual_name = table.get('actual_table_name', odoo_name)

        valid_tables.add(odoo_name)
        valid_tables.add(actual_name)

        table_columns[odoo_name] = []
        table_columns[actual_name] = []

        # Store relationships for validation
        relationships[odoo_name] = table.get('relationships', [])
        relationships[actual_name] = table.get('relationships', [])

        for col in table['columns']:
            col_name = col['name']
            valid_columns.add(col_name)
            table_columns[odoo_name].append(col_name)
            table_columns[actual_name].append(col_name)

    return valid_columns, valid_tables, table_columns, relationships


def validate_sql_against_schema(sql: str, schema: dict) -> Tuple[bool, Optional[str]]:
    """
    COMPREHENSIVE SQL VALIDATION - Multiple layers of security and validation:
    1. Security guard rails (prevent dangerous operations)
    2. Column-to-table relationship validation
    3. Foreign key relationship checking
    4. Schema compliance verification
    """
    valid_columns, valid_tables, table_columns, relationships = get_valid_columns_and_tables(schema)

    # ============================================
    # LAYER 1: SECURITY GUARD RAILS
    # ============================================

    # Block dangerous SQL operations (only SELECT allowed)
    dangerous_keywords = ['drop', 'delete', 'truncate', 'update', 'insert', 'alter', 'create',
                         'grant', 'revoke', 'exec', 'execute', 'into', 'outfile']
    sql_lower = sql.lower()

    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', sql_lower):
            return False, f"🛡️ Security: '{keyword.upper()}' operations not allowed. Only SELECT queries permitted."

    # Check for SQL injection patterns
    injection_patterns = [
        r';.*drop', r';.*delete', r';.*update',
        r'--', r'/\*', r'\*/', r'xp_', r'sp_',
        r'exec\s*\(', r'execute\s*\('
    ]

    for pattern in injection_patterns:
        if re.search(pattern, sql_lower):
            return False, "🛡️ Security: Query contains suspicious patterns. Please use standard SELECT queries."

    # Limit query complexity (max 5 JOINs)
    join_count = len(re.findall(r'\bjoin\b', sql_lower))
    if join_count > 5:
        return False, f"⚠️ Query too complex: {join_count} JOINs found. Maximum 5 allowed for safety."

    # ============================================
    # LAYER 2: SQL KEYWORDS (not columns)
    # ============================================
    sql_keywords = {
        'select', 'from', 'where', 'group', 'order', 'limit', 'offset',
        'join', 'inner', 'left', 'right', 'outer', 'on', 'as', 'and', 'or',
        'having', 'distinct', 'count', 'sum', 'avg', 'max', 'min', 'case',
        'when', 'then', 'else', 'end', 'between', 'like', 'in', 'not', 'is',
        'null', 'asc', 'desc', 'by', 'all', 'exists', 'intersect',
        'with', 'over', 'partition', 'row_number', 'cast', 'numeric', 'interval',
        'date_trunc', 'current_date', 'round', 'cte', 'sub', 'values'
    }

    # ============================================
    # LAYER 3: BUILD TABLE ALIAS MAP
    # ============================================
    # Map aliases to actual table names for column validation
    alias_map = {}  # alias -> table_name

    # Pattern: "FROM/JOIN table_name [AS] alias"
    from_join_pattern = r'\b(?:from|join)\s+(\w+)(?:\s+as\s+|\s+)(\w+)'
    matches = re.findall(from_join_pattern, sql_lower)

    for table_name, alias in matches:
        if alias not in sql_keywords:
            alias_map[alias] = table_name

    # Also handle direct table references (no alias)
    direct_pattern = r'\b(?:from|join)\s+(\w+)(?:\s+(?:on|where|group|order|limit|$))'
    for table_name in re.findall(direct_pattern, sql_lower):
        if table_name in valid_tables:
            alias_map[table_name] = table_name

    # ============================================
    # LAYER 4: VALIDATE COLUMN-TO-TABLE RELATIONSHIPS
    # ============================================
    column_refs = re.findall(r'\b(\w+)\.(\w+)\b', sql_lower)

    errors = []

    for table_alias, col_name in column_refs:
        if col_name in sql_keywords:
            continue

        # Get actual table name from alias
        actual_table = alias_map.get(table_alias)

        if not actual_table:
            # Alias not found - might be in subquery, skip
            continue

        # Check if column exists in schema
        if col_name not in valid_columns:
            similar = [c for c in valid_columns if col_name in c or c in col_name]
            error_msg = f"Column '{col_name}' does not exist in schema."
            if similar:
                error_msg += f" Did you mean: {', '.join(similar[:3])}?"
            errors.append(error_msg)
            continue

        # CRITICAL: Verify column belongs to THIS specific table
        if actual_table in table_columns:
            if col_name not in table_columns[actual_table]:
                error_msg = f"❌ Column '{col_name}' does NOT belong to table '{actual_table}'."
                # Find which tables actually have this column
                correct_tables = [t for t, cols in table_columns.items() if col_name in cols]
                if correct_tables:
                    error_msg += f"\n   ✅ '{col_name}' exists in: {', '.join(correct_tables[:3])}"
                errors.append(error_msg)

    if errors:
        return False, "Schema validation failed:\n" + "\n".join(f"  • {e}" for e in errors)

    # ============================================
    # LAYER 5: VALIDATE FOREIGN KEY RELATIONSHIPS IN JOINS
    # ============================================
    join_conditions = re.findall(r'on\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', sql_lower)

    relationship_warnings = []

    for left_alias, left_col, right_alias, right_col in join_conditions:
        left_table = alias_map.get(left_alias)
        right_table = alias_map.get(right_alias)

        if not left_table or not right_table:
            continue

        # Check if this JOIN uses a documented foreign key relationship
        valid_join = False

        # Check both directions
        if left_table in relationships:
            for rel in relationships[left_table]:
                fk = rel.get('foreign_key', '')
                ref = rel.get('references', '')
                if (fk == left_col or fk == right_col) and (left_col in ref or right_col in ref):
                    valid_join = True
                    break

        if right_table in relationships:
            for rel in relationships[right_table]:
                fk = rel.get('foreign_key', '')
                ref = rel.get('references', '')
                if (fk == left_col or fk == right_col) and (left_col in ref or right_col in ref):
                    valid_join = True
                    break

        # Common valid patterns (id columns are usually safe)
        if 'id' in [left_col, right_col]:
            valid_join = True

        if not valid_join:
            relationship_warnings.append(
                f"⚠️ JOIN {left_table}.{left_col} = {right_table}.{right_col} may not follow schema relationships"
            )

    # Log warnings but don't block (some valid joins might not be documented)
    # In production, you might want to be stricter
    if relationship_warnings:
        # For now, just pass - but these could be logged
        pass

    return True, None


def is_business_query(question: str) -> bool:
    """Check if query is related to sales/business data"""
    business_keywords = [
        'order', 'sale', 'customer', 'product', 'revenue', 'price', 'team',
        'salesperson', 'contact', 'purchase', 'invoice', 'payment', 'quantity',
        'total', 'average', 'top', 'best', 'worst', 'most', 'least', 'show',
        'list', 'find', 'what', 'how many', 'which', 'who', 'when', 'each'
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in business_keywords)


def validate_query(question: str) -> Tuple[bool, Optional[str]]:
    """Validate if query is appropriate for sales order data"""
    non_business_terms = [
        'president', 'weather', 'news', 'movie', 'sports', 'game',
        'celebrity', 'politics', 'capital', 'country', 'wikipedia'
    ]

    question_lower = question.lower()
    if any(term in question_lower for term in non_business_terms):
        return False, "This appears to be a general knowledge question. Please ask questions about sales orders, customers, products, or revenue."

    if not is_business_query(question):
        return False, "Please ask questions related to sales orders, customers, products, or business metrics."

    return True, None


def excel_date_to_datetime(excel_date):
    """Convert Excel serial date to datetime"""
    try:
        return datetime(1899, 12, 30) + pd.Timedelta(days=excel_date)
    except:
        return None


def datetime_to_excel_date(dt: datetime) -> float:
    """Convert datetime to Excel serial date"""
    delta = dt - datetime(1899, 12, 30)
    return delta.days + delta.seconds / 86400.0


def convert_dates_in_sql(sql: str) -> str:
    """
    Convert human-readable date strings in SQL to Excel serial numbers
    Handles formats: 'YYYY-MM-DD', 'DD-MM-YYYY', 'MM/DD/YYYY'
    """
    # Pattern to match date strings in various formats
    date_patterns = [
        (r"'(\d{4})-(\d{1,2})-(\d{1,2})'", '%Y-%m-%d'),  # YYYY-MM-DD
        (r"'(\d{1,2})-(\d{1,2})-(\d{4})'", '%d-%m-%Y'),  # DD-MM-YYYY (European)
        (r"'(\d{1,2})/(\d{1,2})/(\d{4})'", '%m/%d/%Y'),  # MM/DD/YYYY (US)
    ]

    converted_sql = sql

    for pattern, date_format in date_patterns:
        matches = re.findall(pattern, converted_sql)
        for match in matches:
            try:
                # Reconstruct the date string based on format
                if date_format == '%Y-%m-%d':
                    date_str = f"{match[0]}-{match[1]}-{match[2]}"
                elif date_format == '%d-%m-%Y':
                    date_str = f"{match[0]}-{match[1]}-{match[2]}"
                elif date_format == '%m/%d/%Y':
                    date_str = f"{match[0]}/{match[1]}/{match[2]}"

                dt = datetime.strptime(date_str, date_format)
                excel_date = datetime_to_excel_date(dt)

                # Find and replace the original date string
                if '-' in pattern:
                    old_date = f"'{match[0]}-{match[1]}-{match[2]}'"
                else:
                    old_date = f"'{match[0]}/{match[1]}/{match[2]}'"

                converted_sql = converted_sql.replace(old_date, str(excel_date), 1)
            except ValueError:
                # If parsing fails, leave it as is
                continue

    return converted_sql


def get_example_queries() -> List[Dict[str, str]]:
    """
    Comprehensive few-shot examples - ALL 10 client queries plus extras
    Uses Odoo table names (will be mapped to actual names during execution)
    """
    return [
        {
            "question": "List the top 5 selling products within a given date range, ranked by total quantity sold",
            "sql": """SELECT 
    sol.product_id,
    pp.product_tmpl_id,
    pp.name AS product_name,
    SUM(sol.product_uom_qty) AS total_quantity_sold
FROM sale_order_line AS sol
JOIN product_product AS pp ON pp.id = sol.product_id
JOIN sale_order AS so ON so.id = sol.order_id
WHERE so.date_order BETWEEN '2025-01-01' AND '2025-12-31'
GROUP BY sol.product_id, pp.product_tmpl_id, pp.name
ORDER BY total_quantity_sold DESC
LIMIT 5"""
        },
        {
            "question": "For each customer, find the product they've purchased most frequently (highest total quantity)",
            "sql": """SELECT 
    so.partner_id,
    rp.name AS customer_name,
    sol.product_id,
    pp.product_tmpl_id,
    pp.name AS product_name,
    SUM(sol.product_uom_qty) AS total_quantity
FROM sale_order_line sol
JOIN sale_order so ON so.id = sol.order_id
JOIN product_product pp ON pp.id = sol.product_id
LEFT JOIN res_partner rp ON rp.id = so.partner_id
GROUP BY so.partner_id, rp.name, sol.product_id, pp.product_tmpl_id, pp.name
HAVING SUM(sol.product_uom_qty) = (
    SELECT MAX(total_qty)
    FROM (
        SELECT 
            sol2.product_id,
            SUM(sol2.product_uom_qty) AS total_qty
        FROM sale_order_line sol2
        JOIN sale_order so2 ON so2.id = sol2.order_id
        WHERE so2.partner_id = so.partner_id
        GROUP BY sol2.product_id
    ) sub
)"""
        },
        {
            "question": "Calculate the average order size in quantity of items per salesperson for a given time range",
            "sql": """SELECT
    so.user_id,
    ru.name AS salesperson,
    AVG(order_qty) AS avg_order_size_qty
FROM (
    SELECT
        so.id AS order_id,
        so.user_id,
        SUM(sol.product_uom_qty) AS order_qty
    FROM sale_order_line sol
    JOIN sale_order so ON so.id = sol.order_id
    WHERE so.date_order >= '2025-01-01' AND so.date_order < '2026-01-01'
    GROUP BY so.id, so.user_id
) AS per_order
LEFT JOIN res_users ru ON ru.id = per_order.user_id
GROUP BY per_order.user_id, ru.name
ORDER BY avg_order_size_qty DESC"""
        },
        {
            "question": "Determine the product with the highest average quantity per order across all customers",
            "sql": """SELECT 
    sol.product_id,
    pp.product_tmpl_id,
    pp.name AS product_name,
    AVG(sol.product_uom_qty) AS avg_quantity_per_order
FROM sale_order_line AS sol
JOIN product_product AS pp ON pp.id = sol.product_id
JOIN sale_order AS so ON so.id = sol.order_id
GROUP BY sol.product_id, pp.product_tmpl_id, pp.name
ORDER BY avg_quantity_per_order DESC
LIMIT 1"""
        },
        {
            "question": "Show the top customers by total quantity of products ordered, regardless of order value",
            "sql": """SELECT 
    so.partner_id,
    rp.name AS customer_name,
    SUM(sol.product_uom_qty) AS total_quantity_ordered
FROM sale_order_line AS sol
JOIN sale_order AS so ON so.id = sol.order_id
LEFT JOIN res_partner rp ON rp.id = so.partner_id
GROUP BY so.partner_id, rp.name
ORDER BY total_quantity_ordered DESC
LIMIT 10"""
        },
        {
            "question": "Compare sales quantity distribution between two specific sales teams over a given period",
            "sql": """SELECT 
    so.team_id,
    ct.name AS team_name,
    SUM(sol.product_uom_qty) AS total_quantity_sold
FROM sale_order_line AS sol
JOIN sale_order AS so ON so.id = sol.order_id
LEFT JOIN crm_team ct ON ct.id = so.team_id
WHERE so.date_order BETWEEN '2025-01-01' AND '2025-12-31'
  AND so.team_id IN (1, 4)
GROUP BY so.team_id, ct.name
ORDER BY total_quantity_sold DESC"""
        },
        {
            "question": "Find the most frequently ordered product per sales team for a given time range",
            "sql": """WITH team_product AS (
    SELECT
        so.team_id,
        ct.name AS team_name,
        sol.product_id,
        pp.product_tmpl_id,
        pp.name AS product_name,
        SUM(sol.product_uom_qty) AS total_quantity
    FROM sale_order_line sol
    JOIN sale_order so ON so.id = sol.order_id
    JOIN product_product pp ON pp.id = sol.product_id
    LEFT JOIN crm_team ct ON ct.id = so.team_id
    WHERE so.date_order BETWEEN '2025-01-01' AND '2025-12-31'
    GROUP BY so.team_id, ct.name, sol.product_id, pp.product_tmpl_id, pp.name
),
ranked AS (
    SELECT
        team_id,
        team_name,
        product_id,
        product_tmpl_id,
        product_name,
        total_quantity,
        ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY total_quantity DESC) AS rn
    FROM team_product
)
SELECT team_id, team_name, product_id, product_tmpl_id, product_name, total_quantity
FROM ranked
WHERE rn = 1"""
        },
        {
            "question": "List the average unit price per product and highlight any product where the price varies by more than 10% across orders",
            "sql": """WITH product_prices AS (
    SELECT
        sol.product_id,
        pp.product_tmpl_id,
        pp.name AS product_name,
        AVG(sol.price_unit) AS avg_unit_price,
        MAX(sol.price_unit) AS max_price,
        MIN(sol.price_unit) AS min_price
    FROM sale_order_line sol
    JOIN product_product pp ON pp.id = sol.product_id
    JOIN sale_order so ON so.id = sol.order_id
    GROUP BY sol.product_id, pp.product_tmpl_id, pp.name
)
SELECT
    product_id,
    product_tmpl_id,
    product_name,
    avg_unit_price,
    max_price,
    min_price,
    ROUND(((max_price - min_price) / avg_unit_price) * 100, 2) AS price_variation_percent
FROM product_prices
WHERE ((max_price - min_price) / avg_unit_price) * 100 > 10
ORDER BY price_variation_percent DESC"""
        },
        {
            "question": "Calculate the total number of unique products sold by each salesperson during the last quarter",
            "sql": """SELECT                                   
    so.user_id,
    ru.name AS salesperson,
    rp.name AS salesperson_partner,
    COUNT(DISTINCT sol.product_id) AS unique_products_sold
FROM sale_order_line sol
JOIN sale_order so ON so.id = sol.order_id
LEFT JOIN res_users ru ON ru.id = so.user_id
LEFT JOIN res_partner rp ON rp.id = ru.partner_id
WHERE so.date_order >= '2024-10-01' AND so.date_order < '2025-01-01'
GROUP BY so.user_id, ru.name, rp.name
ORDER BY unique_products_sold DESC"""
        },
        {
            "question": "Determine which customers purchase the most diverse product mix (most unique products ordered)",
            "sql": """SELECT
    so.partner_id,
    rp.name AS customer_name,
    COUNT(DISTINCT sol.product_id) AS unique_products_ordered
FROM sale_order_line sol
JOIN sale_order so ON so.id = sol.order_id
LEFT JOIN res_partner rp ON rp.id = so.partner_id
GROUP BY so.partner_id, rp.name
ORDER BY unique_products_ordered DESC
LIMIT 10"""
        },
        {
            "question": "What are the top 5 products by revenue?",
            "sql": """SELECT 
    pp.name AS product,
    pp.product_tmpl_id,
    SUM(sol.price_total) AS total_revenue
FROM sale_order_line sol
JOIN product_product pp ON sol.product_id = pp.id
GROUP BY pp.name, pp.product_tmpl_id
ORDER BY total_revenue DESC
LIMIT 5"""
        },
        {
            "question": "Show all orders from a specific customer",
            "sql": """SELECT 
    so.id,
    so.name AS order_ref,
    rp.name AS customer,
    so.date_order
FROM sale_order so
JOIN res_partner rp ON so.partner_id = rp.id
WHERE rp.name LIKE '%LightsUp%'
ORDER BY so.date_order DESC"""
        },
        {
            "question": "Which sales team generated the most revenue?",
            "sql": """SELECT 
    ct.name AS team,
    SUM(sol.price_total) AS total_revenue
FROM sale_order_line sol
JOIN sale_order so ON sol.order_id = so.id
JOIN crm_team ct ON so.team_id = ct.id
GROUP BY ct.name
ORDER BY total_revenue DESC
LIMIT 5"""
        },
        {
            "question": "What is the average order value per customer?",
            "sql": """SELECT 
    rp.name AS customer,
    AVG(order_total) AS avg_order_value
FROM (
    SELECT 
        so.partner_id,
        so.id AS order_id,
        SUM(sol.price_total) AS order_total
    FROM sale_order_line sol
    JOIN sale_order so ON sol.order_id = so.id
    GROUP BY so.partner_id, so.id
) AS order_totals
JOIN res_partner rp ON order_totals.partner_id = rp.id
GROUP BY rp.name
ORDER BY avg_order_value DESC
LIMIT 10"""
        }
    ]


def generate_sql_with_validation(question: str, schema: dict, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate SQL using GPT with validation
    Maps Odoo table names to actual database names
    """
    try:
        client = OpenAI(api_key=api_key)

        # Build schema context with BOTH Odoo and actual table names
        schema_context = "DATABASE SCHEMA:\n"
        schema_context += "Note: Use Odoo table names (res_partner, crm_team, res_users, product_product) in your queries.\n"
        schema_context += "They will be automatically mapped to actual database tables.\n\n"

        for table in schema['tables']:
            odoo_name = table['table_name']
            actual_name = table.get('actual_table_name', odoo_name)
            schema_context += f"\nTable: {odoo_name}"
            if actual_name != odoo_name:
                schema_context += f" (maps to: {actual_name})"
            schema_context += f"\nDescription: {table.get('description', '')}\n"
            schema_context += "Columns:\n"
            for col in table['columns']:
                col_desc = f"  - {col['name']} ({col['type']})"
                if col.get('description'):
                    col_desc += f": {col['description']}"
                schema_context += col_desc + "\n"

        # Add important notes
        if 'important_notes' in schema:
            schema_context += "\nIMPORTANT NOTES:\n"
            for key, value in schema['important_notes'].items():
                if isinstance(value, dict):
                    schema_context += f"\n{key}:\n"
                    for k, v in value.items():
                        schema_context += f"  {k}: {v}\n"
                else:
                    schema_context += f"- {key}: {value}\n"

        # Build few-shot examples
        examples_text = "\nEXAMPLE QUERIES (use these as reference):\n\n"
        for i, example in enumerate(get_example_queries(), 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Question: {example['question']}\n"
            examples_text += f"SQL:\n{example['sql']}\n\n"

        system_prompt = f"""You are an expert SQL query generator for an Odoo sales database.

{schema_context}

{examples_text}

CRITICAL RULES:
1. Use ONLY table and column names from the schema above
2. Use Odoo table names (res_partner, crm_team, res_users, product_product, etc.)
3. Always include LEFT JOIN for name columns to get readable results
4. For quantity-based queries, use SUM(sol.product_uom_qty)
5. For revenue queries, use SUM(sol.price_total)
6. Date filters: Use so.date_order BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD' format
   - Dates will be automatically converted to Excel serial numbers
   - Accept DD-MM-YYYY or MM/DD/YYYY formats as well
7. Always include product_tmpl_id when selecting from product_product
8. res_users and crm_team have NO direct relationship - use sale_order as bridge
9. When getting salesperson names, join res_users to res_partner via partner_id
10. Queries are valid even if they return no results (empty dataset is OK)

Generate ONLY the SQL query, no explanations."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate SQL for: {question}"}
            ],
            temperature=0.1,
            max_tokens=800
        )

        sql = response.choices[0].message.content.strip()

        # Clean up the SQL
        sql = sql.replace('```sql', '').replace('```', '').strip()

        # Validate against schema (before mapping and date conversion)
        is_valid, validation_error = validate_sql_against_schema(sql, schema)
        if not is_valid:
            return None, f"Schema validation failed: {validation_error}"

        # Convert date strings to Excel serial numbers
        sql = convert_dates_in_sql(sql)

        # Map Odoo table names to actual database table names
        mapped_sql = map_table_names_in_sql(sql)

        return mapped_sql, None

    except Exception as e:
        return None, f"Error generating SQL: {str(e)}"


def execute_query(sql: str, conn) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute SQL query and return results"""
    try:
        df = pd.read_sql_query(sql, conn)
        return df, None
    except Exception as e:
        return None, f"Query execution error: {str(e)}"


def generate_natural_language_answer(question: str, df: pd.DataFrame, api_key: str) -> str:
    """Generate natural language answer from query results"""
    try:
        client = OpenAI(api_key=api_key)

        if df is not None and len(df) > 0:
            data_summary = f"Query returned {len(df)} row(s).\n\n"
            data_summary += "Sample data:\n"
            for idx, row in df.head(10).iterrows():
                data_summary += f"Row {idx+1}: "
                parts = []
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        if isinstance(val, float):
                            if any(x in col.lower() for x in ['revenue', 'total', 'price', 'sales']):
                                val = f"${val:,.2f}"
                            else:
                                val = f"{val:,.2f}"
                        parts.append(f"{col}={val}")
                data_summary += ", ".join(parts) + "\n"
        else:
            data_summary = "No results found."

        prompt = f"""You are a business analyst. Answer the question clearly based on the data.

Question: {question}

Data:
{data_summary}

Provide a clear answer in 2-3 sentences. Use natural language. Format numbers with $ and commas where appropriate."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst who explains data clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except:
        if df is not None and len(df) > 0:
            return f"Found {len(df)} result(s)."
        return "No results found."


def format_dataframe_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe for display"""
    display_df = df.copy()

    for col in display_df.columns:
        if 'date' in col.lower() and display_df[col].dtype in ['float64', 'float32']:
            try:
                display_df[col] = display_df[col].apply(
                    lambda x: excel_date_to_datetime(x).strftime('%Y-%m-%d') if pd.notna(x) else None
                )
            except:
                pass
        elif any(x in col.lower() for x in ['revenue', 'total', 'price', 'sales']) and display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else None)

    return display_df


def get_stats(conn):
    """Get database statistics"""
    try:
        cursor = conn.cursor()

        stats = {}
        for table in ['res_partner', 'crm_team', 'res_users', 'product_product', 'sale_order', 'sale_order_line']:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            except:
                stats[table] = 0

        try:
            cursor.execute("SELECT SUM(price_total) FROM sale_order_line")
            result = cursor.fetchone()
            stats['revenue'] = result[0] if result[0] else 0
        except:
            stats['revenue'] = 0

        try:
            cursor.execute("SELECT AVG(order_total) FROM (SELECT order_id, SUM(price_total) as order_total FROM sale_order_line GROUP BY order_id)")
            result = cursor.fetchone()
            stats['avg_order'] = result[0] if result[0] else 0
        except:
            stats['avg_order'] = 0

        return stats
    except Exception as e:
        return {}


def main():
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    st.title("📊 Sales Analytics Platform")
    st.caption("Ask questions about your sales data in plain English")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")

        st.divider()
        st.header("📊 Database Overview")

        conn_stats = get_db_connection()
        if conn_stats:
            stats = get_stats(conn_stats)
            if stats:
                st.metric("Total Customers", f"{stats.get('res_partner', 0):,}")
                st.metric("Total Orders", f"{stats.get('sale_order', 0):,}")
                st.metric("Total Revenue", f"${stats.get('revenue', 0):,.0f}")

        st.divider()
        st.header("ℹ️ System Info")
        st.caption("Table Mapping:")
        st.caption("• res_partner → contact")
        st.caption("• crm_team → sales_team")
        st.caption("• res_users → user")
        st.caption("• product_product → product_variant")


    schema = load_schema()
    conn = get_db_connection()

    if not schema or not conn:
        st.error("System initialization failed.")
        return

    st.subheader("Ask a Question")

    # Quick buttons
    st.write("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Top 5 Products", use_container_width=True):
            st.session_state.current_query = "What are the top 5 products by revenue?"
            st.rerun()

    with col2:
        if st.button("📈 Top by Quantity", use_container_width=True):
            st.session_state.current_query = "List top 5 products by quantity sold in 2025"
            st.rerun()

    with col3:
        if st.button("👥 Customer Preference", use_container_width=True):
            st.session_state.current_query = "For each customer, what product do they buy most?"
            st.rerun()

    with col4:
        if st.button("💰 Diverse Customers", use_container_width=True):
            st.session_state.current_query = "Which customers buy the most diverse products?"
            st.rerun()

    st.divider()

    query = st.text_area(
        "Or type your own question:",
        value=st.session_state.current_query,
        height=80,
        placeholder="e.g., Which products generated the most revenue?"
    )

    if query != st.session_state.current_query:
        st.session_state.current_query = query

    if st.button("▶️ Get Answer", type="primary"):
        if not api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar")
            return

        if not query or len(query.strip()) < 5:
            st.warning("⚠️ Please enter a question (at least 5 characters)")
            return

        is_valid, validation_msg = validate_query(query)
        if not is_valid:
            st.error(f"🚫 {validation_msg}")
            return

        # Generate and VALIDATE SQL
        with st.spinner("🤖 Processing your question..."):
            sql, error = generate_sql_with_validation(query, schema, api_key)

        if error:
            st.error(f"❌ {error}")
            st.info("💡 Try rephrasing your question or use one of the quick questions above")
            return

        # Execute (only if validation passed)
        with st.spinner("📊 Executing query..."):
            result_df, error = execute_query(sql, conn)

        if error:
            st.error(f"❌ {error}")
            with st.expander("🔍 View SQL"):
                st.code(sql, language='sql')
            return

        # Generate answer
        with st.spinner("✍️ Preparing answer..."):
            nl_answer = generate_natural_language_answer(query, result_df, api_key)

        st.success("✅ Query executed successfully!")

        st.subheader("💬 Answer")
        st.info(nl_answer)

        # ALWAYS show SQL query (even when no data)
        with st.expander("🔧 View SQL Query"):
            st.code(sql, language='sql')

        if result_df is not None and len(result_df) > 0:
            st.divider()
            st.subheader("📋 Data")

            display_df = format_dataframe_display(result_df)
            st.dataframe(display_df, use_container_width=True, height=400)

            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.info("ℹ️ No data found for this query")


if __name__ == "__main__":
    main()