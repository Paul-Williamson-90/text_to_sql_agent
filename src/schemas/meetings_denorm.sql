with firms_discussed_name_join as (
    SELECT 
        mf.meeting_id, 
        string_agg(f.name, ', ') AS firms_discussed_names
    FROM meeting_firms mf
    LEFT JOIN firms f
    on mf.firm_id = f.firm_id
    GROUP BY mf.meeting_id
),

firms_discussed_sector_join as (
    SELECT 
        mf.meeting_id, 
        string_agg(f.sector, ', ') AS firms_discussed_sectors
    FROM meeting_firms mf
    LEFT JOIN firms f
    on mf.firm_id = f.firm_id
    GROUP BY mf.meeting_id
),

contacts_attended_join as (
    SELECT
        cm.meeting_id,
        string_agg(c.contact_name, ', ') AS contacts_attended
    FROM contact_meetings cm
    LEFT JOIN (
        SELECT
            c.contact_id,
            CONCAT(c.name, ' (', f.name, ')') AS contact_name
        FROM contacts c
        LEFT JOIN firms f
        on c.firm_id = f.firm_id
    ) c
    on cm.contact_id = c.contact_id
    GROUP BY cm.meeting_id
),

employees_attended_join as (
    SELECT
        em.meeting_id,
        string_agg(e.name, ', ') AS employees_attended
    FROM employee_meetings em
    LEFT JOIN employees e
    on em.employee_id = e.employee_id
    GROUP BY em.meeting_id
),

firm_attended_join as (
    SELECT
        mf.meeting_id,
        f.name AS firm_attended
    FROM meetings mf
    LEFT JOIN firms f
    on mf.firm_attended_id = f.firm_id
)

SELECT
    m.meeting_id,
    m.beam_id,
    m.date,
    m.title,
    m.content,
    fa.firm_attended as firm_attended_name,
    f.firms_discussed_names,
    s.firms_discussed_sectors,
    c.contacts_attended,
    e.employees_attended
FROM meetings m
LEFT JOIN firms_discussed_name_join f
on m.meeting_id = f.meeting_id
LEFT JOIN firms_discussed_sector_join s
on m.meeting_id = s.meeting_id
LEFT JOIN contacts_attended_join c
on m.meeting_id = c.meeting_id
LEFT JOIN employees_attended_join e
on m.meeting_id = e.meeting_id
LEFT JOIN firm_attended_join fa
on m.meeting_id = fa.meeting_id