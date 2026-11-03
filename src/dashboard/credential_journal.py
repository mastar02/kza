"""Closed credential intents and durable, secret-free terminal receipts."""

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from uuid import UUID

from src.dashboard.authority import DashboardRole, _identifier, _timestamp
from src.dashboard.evaluation_grants import EvaluationGrantSelection

MAX_CREDENTIAL_ATTEMPTS = 4096
MAX_CREDENTIAL_REVISION = (1<<63)-1


class CredentialAction(StrEnum):
    ISSUE = 'issue'
    REVISE = 'revise'
    REVOKE = 'revoke'


class CredentialOutcome(StrEnum):
    COMMITTED = 'committed'
    NO_EFFECT = 'no_effect'


REASONS = frozenset({'applied','actor_retired','scope_unavailable','grant_invalid',
    'account_unavailable','writer_busy','target_missing','revision_conflict','capacity'})


def canonical_attempt(value):
    if type(value) is not str or str(UUID(value))!=value:
        raise ValueError('invalid_credential_attempt')
    return value


def digest_value(value):
    if type(value) is not str or len(value)!=64 or any(char not in '0123456789abcdef' for char in value):
        raise ValueError('invalid_credential_command_digest')
    return value


@dataclass(frozen=True,repr=False)
class CredentialGrantSpec:
    role: DashboardRole
    expires_at: float
    user_id: str | None = None
    selections: tuple[EvaluationGrantSelection,...] = ()
    subject: str | None = None

    def __post_init__(self):
        if type(self.role) is not DashboardRole:
            raise ValueError('invalid_credential_role')
        _timestamp(self.expires_at)
        if self.user_id is not None:
            _identifier(self.user_id)
            if self.role is not DashboardRole.HOME:
                raise ValueError('invalid_credential_binding')
        if self.subject is not None:
            _identifier(self.subject)
        if (type(self.selections) is not tuple or len(self.selections)>64
                or any(type(value) is not EvaluationGrantSelection for value in self.selections)
                or len({value.fingerprint_id for value in self.selections})!=len(self.selections)
                or (self.role is DashboardRole.EVALUATOR)!=bool(self.selections)):
            raise ValueError('invalid_credential_selections')

    def requested(self):
        return {'role':self.role.value,'expires_at':self.expires_at,'user_id':self.user_id,
            'subject':self.subject,'selections':[{'fingerprint_id':value.fingerprint_id,
                'modalities':sorted(mode.value for mode in value.modalities),'expires_at':value.expires_at}
                for value in sorted(self.selections,key=lambda value:value.fingerprint_id)]}


@dataclass(frozen=True,repr=False)
class CredentialCommand:
    request_id: str
    action: CredentialAction
    credential_id: str | None = None
    expected_revision: int | None = None
    grant: CredentialGrantSpec | None = None

    def __post_init__(self):
        canonical_attempt(self.request_id)
        if type(self.action) is not CredentialAction:
            raise ValueError('invalid_credential_action')
        if self.action is CredentialAction.ISSUE:
            if (self.credential_id is not None or self.expected_revision is not None
                    or type(self.grant) is not CredentialGrantSpec or self.grant.subject is None):
                raise ValueError('invalid_credential_issue')
        else:
            _identifier(self.credential_id)
            if (type(self.expected_revision) is not int
                    or not 1<=self.expected_revision<=MAX_CREDENTIAL_REVISION-(self.action is CredentialAction.REVISE)):
                raise ValueError('invalid_credential_revision')
            if self.action is CredentialAction.REVISE:
                if type(self.grant) is not CredentialGrantSpec or self.grant.subject is not None:
                    raise ValueError('invalid_credential_revision_grant')
            elif self.grant is not None:
                raise ValueError('invalid_credential_revocation')

    @property
    def target(self):
        return self.request_id if self.action is CredentialAction.ISSUE else self.credential_id

    @property
    def digest(self):
        value = {'schema':1,'action':self.action.value,'credential_id':self.credential_id,
            'expected_revision':self.expected_revision,'grant':None if self.grant is None else self.grant.requested()}
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True,repr=False)
class CredentialReceipt:
    request_id: str
    actor_key: str = field(repr=False)
    command_digest: str = field(repr=False)
    action: CredentialAction
    credential_id: str
    expected_revision: int | None
    outcome: CredentialOutcome
    result_revision: int | None
    reason: str
    completed_at: float

    def __post_init__(self):
        canonical_attempt(self.request_id)
        digest_value(self.actor_key)
        digest_value(self.command_digest)
        _identifier(self.credential_id)
        if (type(self.action) is not CredentialAction or type(self.outcome) is not CredentialOutcome
                or type(self.reason) is not str or self.reason not in REASONS
                or (self.outcome is CredentialOutcome.COMMITTED)!=(self.reason=='applied')
                or self.expected_revision is not None and (type(self.expected_revision) is not int or not 1<=self.expected_revision<=MAX_CREDENTIAL_REVISION)
                or self.result_revision is not None and (type(self.result_revision) is not int or not 1<=self.result_revision<=MAX_CREDENTIAL_REVISION)
                or (self.action is CredentialAction.ISSUE)!=(self.expected_revision is None)
                or self.outcome is CredentialOutcome.COMMITTED and self.result_revision is None
                or self.outcome is CredentialOutcome.NO_EFFECT and self.result_revision is not None
                or self.action is CredentialAction.ISSUE and self.credential_id!=self.request_id
                or self.outcome is CredentialOutcome.COMMITTED and (
                    self.action is CredentialAction.ISSUE and self.result_revision!=1
                    or self.action is CredentialAction.REVISE and self.result_revision!=self.expected_revision+1
                    or self.action is CredentialAction.REVOKE and self.result_revision!=self.expected_revision)):
            raise ValueError('invalid_credential_receipt')
        _timestamp(self.completed_at)

    def public(self):
        return {'request_id':self.request_id,'action':self.action.value,'credential_id':self.credential_id,
            'expected_revision':self.expected_revision,'outcome':self.outcome.value,
            'result_revision':self.result_revision,'reason':self.reason,'completed_at':self.completed_at}

    def row(self):
        return (self.request_id,self.actor_key,self.command_digest,self.action.value,self.credential_id,
            self.expected_revision,self.outcome.value,self.result_revision,self.reason,self.completed_at)


@dataclass(frozen=True,repr=False)
class CredentialCommandResult:
    receipt: CredentialReceipt
    applied: bool
    token: str | None = field(default=None,repr=False)


JOURNAL_DDL = '''CREATE TABLE credential_attempts (
 request_id TEXT PRIMARY KEY, actor_key TEXT NOT NULL, command_digest TEXT NOT NULL,
 kind TEXT NOT NULL, credential_id TEXT NOT NULL, expected_revision INTEGER,
 outcome TEXT NOT NULL, result_revision INTEGER, reason TEXT NOT NULL, completed_at REAL NOT NULL)'''

TABLES = {
    'auth_meta':(('key','TEXT',0,1),('value','TEXT',1,0)),
    'credentials':(('id','TEXT',0,1),('digest','TEXT',1,0),('subject','TEXT',1,0),('role','TEXT',1,0),
        ('user_id','TEXT',0,0),('issued_at','REAL',1,0),('expires_at','REAL',1,0),('revision','INTEGER',1,0),('samples','TEXT',1,0)),
    'sessions':(('digest','TEXT',0,1),('credential','TEXT',1,0),('legacy_digest','TEXT',0,0),
        ('issued_at','REAL',1,0),('expires_at','REAL',1,0),('revision','INTEGER',1,0)),
    'retired_users':(('user_id','TEXT',0,1),('retired_at','REAL',1,0)),
    'credential_attempts':(('request_id','TEXT',0,1),('actor_key','TEXT',1,0),('command_digest','TEXT',1,0),
        ('kind','TEXT',1,0),('credential_id','TEXT',1,0),('expected_revision','INTEGER',0,0),
        ('outcome','TEXT',1,0),('result_revision','INTEGER',0,0),('reason','TEXT',1,0),('completed_at','REAL',1,0)),
}


def validate_table(db,name):
    actual = tuple((row['name'],row['type'].upper(),row['notnull'],row['pk']) for row in db.execute('PRAGMA table_info('+name+')'))
    kind = db.execute("SELECT type FROM sqlite_master WHERE name=?",(name,)).fetchone()
    if kind is None or kind[0]!='table' or actual!=TABLES[name] or tuple(db.execute('PRAGMA foreign_key_list('+name+')')):
        raise ValueError('invalid_dashboard_authority_table')
    if name=='credentials':
        unique_digest = any(row['unique']==1 and row['partial']==0
            and tuple(item[0] for item in db.execute('SELECT name FROM pragma_index_info(?) ORDER BY seqno',(row['name'],)))==('digest',)
            for row in db.execute('PRAGMA index_list(credentials)'))
        if not unique_digest:
            raise ValueError('invalid_dashboard_credential_digest_index')


def read_metadata(db):
    meta = dict(db.execute('SELECT key,value FROM auth_meta'))
    if not {'schema_version','clock_floor','legacy_epoch','legacy_digest'}<=meta.keys():
        raise ValueError('missing_dashboard_authority_metadata')
    floor = float(meta['clock_floor'])
    _timestamp(floor)
    epoch = int(meta['legacy_epoch'])
    if epoch<0 or str(epoch)!=meta['legacy_epoch']:
        raise ValueError('invalid_dashboard_authority_epoch')
    digest = meta['legacy_digest']
    if digest:
        digest_value(digest)
    return floor,epoch,digest or None


def migrate_schema(db,base_schema):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if not tables:
        for statement in base_schema.split(';'):
            if statement.strip():
                db.execute(statement)
        db.execute(JOURNAL_DDL)
        db.executemany('INSERT INTO auth_meta VALUES (?,?)',(
            ('schema_version','3'),('clock_floor','0'),('legacy_epoch','0'),('legacy_digest','')))
    else:
        if 'auth_meta' not in tables:
            raise ValueError('missing_dashboard_authority_metadata')
        validate_table(db,'auth_meta')
        row = db.execute("SELECT value FROM auth_meta WHERE key='schema_version'").fetchone()
        if row is None or row[0] not in {'1','2','3'}:
            raise ValueError('unsupported_dashboard_authority_schema')
        for name in ('credentials','sessions'):
            validate_table(db,name)
        if row[0] in {'2','3'} or 'retired_users' in tables:
            validate_table(db,'retired_users')
        elif row[0]=='1':
            db.execute('CREATE TABLE retired_users (user_id TEXT PRIMARY KEY, retired_at REAL NOT NULL)')
        if row[0]=='3':
            validate_table(db,'credential_attempts')
        elif 'credential_attempts' in tables:
            raise ValueError('partial_dashboard_credential_journal')
        else:
            db.execute(JOURNAL_DDL)
    for name in TABLES:
        validate_table(db,name)
    floor,_,_ = read_metadata(db)
    for name,column in (('credentials','issued_at'),('sessions','issued_at'),
            ('retired_users','retired_at'),('credential_attempts','completed_at')):
        if db.execute('SELECT 1 FROM '+name+' WHERE typeof('+column+") NOT IN ('real','integer') OR "+column+'<0 OR '+column+'>? LIMIT 1',(floor,)).fetchone():
            raise ValueError('invalid_dashboard_authority_history_clock')
    db.execute("INSERT OR REPLACE INTO auth_meta VALUES ('schema_version','3')")


def read_attempts(db):
    validate_table(db,'credential_attempts')
    result = {}
    for row in db.execute('SELECT * FROM credential_attempts LIMIT ?', (MAX_CREDENTIAL_ATTEMPTS+1,)):
        receipt = CredentialReceipt(row['request_id'],row['actor_key'],row['command_digest'],CredentialAction(row['kind']),
            row['credential_id'],row['expected_revision'],CredentialOutcome(row['outcome']),row['result_revision'],
            row['reason'],row['completed_at'])
        result[receipt.request_id] = receipt
    if len(result)>MAX_CREDENTIAL_ATTEMPTS:
        raise ValueError('dashboard_credential_journal_capacity')
    return result


def validate_history(credentials,attempts,floor):
    revisions,revoked = {},set()
    for receipt in attempts.values():
        if receipt.completed_at>floor:
            raise ValueError('invalid_dashboard_credential_history_clock')
        if receipt.outcome is CredentialOutcome.COMMITTED:
            identity = receipt.credential_id
            revisions[identity] = max(revisions.get(identity,0),receipt.result_revision)
            if receipt.action is CredentialAction.REVOKE:
                revoked.add(identity)
    if any(entry.identity in revoked or entry.revision<revisions.get(entry.identity,0)
            for entry in credentials.values()):
        raise ValueError('invalid_dashboard_credential_history')


def insert_receipt(db,receipt):
    db.execute('INSERT INTO credential_attempts VALUES (?,?,?,?,?,?,?,?,?,?)',receipt.row())
