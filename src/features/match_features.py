"""
Skill-gap analysis: compares a resume's text against a job's required skills
to find which skills match and which are missing.
"""

def get_job_skills(job_row):
    """
    Extract a clean list of individual skills from a job's 'Key Skills' column.
    The raw format looks like: "hiring| billing executive| US Healthcare"
    """
    raw = job_row.get("Key Skills", "")
    if not isinstance(raw, str) or not raw.strip():   # handle missing/empty values safely
        return []
    # split on "|", strip whitespace from each piece, drop any empty strings
    skills = [s.strip() for s in raw.split("|") if s.strip()]
    return skills


def skill_gap(resume_text, job_skills):
    """
    Compare a resume's raw text against a list of required job skills.
    Returns (matched_skills, missing_skills) as two separate lists.
    Uses simple case-insensitive substring matching — no NLP needed,
    keeps this fully classical/offline as per project scope.
    """
    resume_lower = resume_text.lower()      # lowercase once, for case-insensitive comparison
    matched = []
    missing = []
    for skill in job_skills:
        if skill.lower() in resume_lower:    # if the skill phrase appears anywhere in the resume
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing