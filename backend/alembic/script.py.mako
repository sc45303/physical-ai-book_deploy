<%!
import re

%>\
<% revision = re.sub(r'[^\w_]', '_', revision) %>\
<% down_revision = re.sub(r'[^\w_]', '_', down_revision) if down_revision else None %>\
"""${message}

Revision ID: ${revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade():
    ${upgrades if upgrades else "pass"}


def downgrade():
    ${downgrades if downgrades else "pass"}