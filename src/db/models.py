from __future__ import annotations

import uuid
from textwrap import dedent

from sqlalchemy import TIMESTAMP, UUID, Column, ForeignKey, String, Table, text
from sqlalchemy.orm import relationship

from src.db.database import Base, engine

contacts_attended_meetings_association = Table(
    "contact_meetings",
    Base.metadata,
    Column("contact_id", UUID(as_uuid=True), ForeignKey("contacts.contact_id")),
    Column("meeting_id", UUID(as_uuid=True), ForeignKey("meetings.meeting_id")),
)

meetings_firms_discussed_association = Table(
    "meeting_firms",
    Base.metadata,
    Column("meeting_id", UUID(as_uuid=True), ForeignKey("meetings.meeting_id")),
    Column("firm_id", UUID(as_uuid=True), ForeignKey("firms.firm_id")),
)

employees_attended_meetings_association = Table(
    "employee_meetings",
    Base.metadata,
    Column("employee_id", UUID(as_uuid=True), ForeignKey("employees.employee_id")),
    Column("meeting_id", UUID(as_uuid=True), ForeignKey("meetings.meeting_id")),
)


class Meetings(Base):
    __tablename__ = "meetings"
    __context_str__ = dedent(
        """
        Table Name: meetings
        Description: A table that stores meeting notes for various meetings that have taken place.
        Columns:
            - meeting_id: UUID, primary_key:
                - Unique identifier for the meeting, this should never be presented to the user.
            - beam_id: UUID
                - Unique identifier for the meeting, this must be presented to the user wherever possible.
            - title: String, not null
                - The title of the meeting.
            - content: String, not null
                - The meeting notes recorded for the meeting.
            - date: TIMESTAMP
                - The date and time the meeting took place.
            - created_at: TIMESTAMP
                - The date and time the row was created in the database.
            - firm_attended_id: UUID, ForeignKey('firms.firm_id')
                - The id of the firm that attended the meeting.
            - firm_attended: relationship('Firms', foreign_keys=[firm_attended_id], back_populates='meetings_attended')
                - The firm that attended the meeting.
            - contacts: relationship('Contacts', secondary=contacts_attended_meetings_association, back_populates='meetings_attended')
                - The external contacts from Firms that attended the meeting.
            - employees: relationship('Employees', secondary=employees_attended_meetings_association, back_populates='meetings_attended')
                - The internal employees that attended the meeting.
            - firms_discussed: relationship('Firms', secondary=meetings_firms_discussed_association, back_populates='meetings_discussed')
                - The firms that were discussed during the meeting.
        """
    )

    meeting_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the meeting, this should never be presented to the user.",
    )
    beam_id = Column(
        UUID(as_uuid=True),
        default=uuid.uuid4,
        comment="Unique identifier for the meeting, this must be presented to the user wherever possible.",
    )
    title = Column(String, nullable=False, comment="The title of the meeting.")
    content = Column(
        String, nullable=False, comment="The meeting notes recorded for the meeting."
    )
    date = Column(
        TIMESTAMP(timezone=True), comment="The date and time the meeting took place."
    )
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=text("now()"),
        comment="The date and time the row was created in the database.",
    )

    firm_attended_id = Column(
        UUID(as_uuid=True),
        ForeignKey("firms.firm_id"),
        comment="The id of the firm that attended the meeting.",
    )
    firm_attended = relationship(
        "Firms", foreign_keys=[firm_attended_id], back_populates="meetings_attended"
    )
    contacts = relationship(
        "Contacts",
        secondary=contacts_attended_meetings_association,
        back_populates="meetings_attended",
    )
    employees = relationship(
        "Employees",
        secondary=employees_attended_meetings_association,
        back_populates="meetings_attended",
    )
    firms_discussed = relationship(
        "Firms",
        secondary=meetings_firms_discussed_association,
        back_populates="meetings_discussed",
    )


class Firms(Base):
    __tablename__ = "firms"
    __context_str__ = dedent(
        """
        Table Name: firms
        Description: A table that stores information on Firms that the company is/has interacted with or discussed in meetings.
        Columns:
            - firm_id: UUID, primary_key:
                - Unique identifier for the firm, this should never be presented to the user.
            - name: String, not null
                - The name of the firm.
            - sector: String
                - The sector the firm operates in.
            - created_at: TIMESTAMP
                - The date and time the row was created in the database.
            - contacts: relationship('Contacts', back_populates='firm')
                - The contacts that work for the firm.
            - meetings_attended: relationship('Meetings', foreign_keys='Meetings.firm_attended_id', back_populates='firm_attended')
                - The meetings that the firm attended.
            - meetings_discussed: relationship('Meetings', secondary=meetings_firms_discussed_association, back_populates='firms_discussed')
                - The meetings where the firm was discussed.
        """
    )

    firm_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the firm, this should never be presented to the user.",
    )
    name = Column(String, nullable=False, comment="The name of the firm.")
    sector = Column(String, comment="The sector the firm operates in.")
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=text("now()"),
        comment="The date and time the row was created in the database.",
    )

    contacts = relationship("Contacts", back_populates="firm")
    meetings_attended = relationship(
        "Meetings",
        foreign_keys="Meetings.firm_attended_id",
        back_populates="firm_attended",
    )
    meetings_discussed = relationship(
        "Meetings",
        secondary=meetings_firms_discussed_association,
        back_populates="firms_discussed",
    )


class Contacts(Base):
    __tablename__ = "contacts"
    __context_str__ = dedent(
        """
        Table Name: contacts
        Description: A table that stores information on external contacts that work at firms the company is/has interacted with or discussed.
        Columns:
            - contact_id: UUID, primary_key:
                - Unique identifier for the contact, this should never be presented to the user.
            - name: String, not null
                - The name of the contact.
            - email: String
                - The email address of the contact.
            - address: String
                - The address of the contact.
            - created_at: TIMESTAMP
                - The date and time the row was created in the database.
            - firm_id: UUID, ForeignKey('firms.firm_id')
                - The id of the firm the contact works for.
            - firm: relationship('Firms', back_populates='contacts')
                - The firm the contact works for.
            - meetings_attended: relationship('Meetings', secondary=contacts_attended_meetings_association, back_populates='contacts_attended')
                - The meetings the contact attended.
        """
    )

    contact_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the contact, this should never be presented to the user.",
    )
    name = Column(String, nullable=False, comment="The name of the contact.")
    email = Column(String, comment="The email address of the contact.")
    address = Column(String, comment="The address of the contact.")
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=text("now()"),
        comment="The date and time the row was created in the database.",
    )

    firm_id = Column(UUID(as_uuid=True), ForeignKey("firms.firm_id"))
    firm = relationship("Firms", back_populates="contacts")
    meetings_attended = relationship(
        "Meetings",
        secondary=contacts_attended_meetings_association,
        back_populates="contacts",
    )


class Employees(Base):
    __tablename__ = "employees"
    __context_str__ = dedent(
        """
        Table Name: employees
        Description: A table that stores information on internal employees for our company.
        Columns:
            - employee_id: UUID, primary_key:
                - Unique identifier for the employee, this should never be presented to the user.
            - name: String, not null
                - The name of the employee.
            - email: String
                - The email address of the employee.
            - created_at: TIMESTAMP
                - The date and time the row was created in the database.
            - meetings_attended: relationship('Meetings', secondary=employees_attended_meetings_association, back_populates='employees')
                - The meetings the employee attended.
        """
    )

    employee_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the employee, this should never be presented to the user.",
    )
    name = Column(String, nullable=False, comment="The name of the employee.")
    email = Column(String, comment="The email address of the employee.")
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=text("now()"),
        comment="The date and time the row was created in the database.",
    )

    meetings_attended = relationship(
        "Meetings",
        secondary=employees_attended_meetings_association,
        back_populates="employees",
    )


Base.metadata.create_all(bind=engine)

"""
# EXAMPLE QUERIES:

## Use-case: Retrieving all meetings attended by a specific firm.
```postgresql
SELECT meetings.meeting_id, meetings.beam_id, meetings.title, meetings.content, meetings.date, meetings.created_at, meetings.firm_attended_id
FROM meetings JOIN firms ON meetings.firm_attended_id = firms.firm_id
WHERE firms.name ILIKE '%Marathon Petroleum%';
```

## Use-case: Retrieving all meetings attended by a specific contact.
```postgresql
SELECT meetings.meeting_id, meetings.beam_id, meetings.title, meetings.content, meetings.date, meetings.created_at, meetings.firm_attended_id
FROM meetings JOIN contact_meetings AS contact_meetings_1 ON meetings.meeting_id = contact_meetings_1.meeting_id JOIN contacts ON contacts.contact_id = contact_meetings_1.contact_id
WHERE contacts.name ILIKE '%John Doe%';
```

## Use-case: Retrieving all meetings attended by a specific employee.
```postgresql
SELECT meetings.meeting_id, meetings.beam_id, meetings.title, meetings.content, meetings.date, meetings.created_at, meetings.firm_attended_id
FROM meetings JOIN employee_meetings AS employee_meetings_1 ON meetings.meeting_id = employee_meetings_1.meeting_id JOIN employees ON employees.employee_id = employee_meetings_1.employee_id
WHERE employees.name ILIKE '%John Doe%';
```

## Use-case: Retrieving all meetings and who attended them (internal employees and external contacts).
```postgresql
SELECT meetings.meeting_id, meetings.beam_id, meetings.title, meetings.content, meetings.date, meetings.created_at, meetings.firm_attended_id, employees.name, contacts.name AS name_1
FROM meetings JOIN employee_meetings AS employee_meetings_1 ON meetings.meeting_id = employee_meetings_1.meeting_id JOIN employees ON employees.employee_id = employee_meetings_1.employee_id JOIN contact_meetings AS contact_meetings_1 ON meetings.meeting_id = contact_meetings_1.meeting_id JOIN contacts ON contacts.contact_id = contact_meetings_1.contact_id;
```

## Use-case: Retrieving all meetings and which firms were discussed in them.
```postgresql
SELECT meetings.meeting_id, meetings.beam_id, meetings.title, meetings.content, meetings.date, meetings.created_at, meetings.firm_attended_id, firms.name
FROM meetings JOIN meeting_firms AS meeting_firms_1 ON meetings.meeting_id = meeting_firms_1.meeting_id JOIN firms ON firms.firm_id = meeting_firms_1.firm_id;
```

## Use-case: Retrieving all meetings between a specific date range that a specific firm that either attended the meeting or was discussed in the meeting.
```postgresql
SELECT meetings.meeting_id, meetings.beam_id, meetings.title, meetings.content, meetings.date, meetings.created_at, meetings.firm_attended_id
FROM meetings
WHERE meetings.date >= '2025-01-15' AND meetings.date <= '2025-02-15' AND ((EXISTS (SELECT 1
FROM firms, meeting_firms
WHERE meetings.meeting_id = meeting_firms.meeting_id
AND firms.firm_id = meeting_firms.firm_id
AND firms.name ILIKE '%company name%'))
OR meetings.firm_attended_id in (SELECT firms.firm_id FROM firms WHERE firms.name ILIKE '%company name%'))
```
"""
