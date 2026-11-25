//! Scoped household lists and the Home Assistant todo integration contract.

use std::future::Future;

use rusqlite::{Connection, OptionalExtension, Row, params};

use crate::{Clock, Error, Household, HouseholdWorker, Result, name_key, query_all, required};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemId(pub i64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct List {
    pub id: ListId,
    pub name: String,
    pub owner: Option<String>,
    pub ha_entity: Option<String>,
    pub revision: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListItem {
    pub id: ItemId,
    pub text: String,
    pub completed: bool,
    pub ha_uid: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListSnapshot {
    pub list: List,
    pub items: Vec<ListItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TodoChange {
    Add {
        uid: String,
        text: String,
        completed: bool,
    },
    Update {
        item: ItemId,
        uid: String,
        text: String,
        completed: bool,
    },
    Remove(ItemId),
}

/// An adapter reconciles a shared list snapshot against its HA todo entity.
/// Remote writes must be idempotent across retries and revision conflicts.
pub trait TodoSync {
    type Error: std::error::Error;

    fn reconcile(
        &self,
        entity_id: &str,
        snapshot: &ListSnapshot,
    ) -> impl Future<Output = std::result::Result<Vec<TodoChange>, Self::Error>> + Send;
}

#[derive(Debug, thiserror::Error)]
pub enum TodoSyncError<E: std::error::Error> {
    #[error(transparent)]
    Household(#[from] Error),
    #[error("Home Assistant todo sync failed: {0}")]
    Adapter(#[source] E),
}

const LIST_COLUMNS: &str = "id, name, owner, ha_entity, revision";
const ITEM_COLUMNS: &str = "id, text, completed, ha_uid";

fn read_list(row: &Row<'_>) -> rusqlite::Result<List> {
    Ok(List {
        id: ListId(row.get(0)?),
        name: row.get(1)?,
        owner: row.get(2)?,
        ha_entity: row.get(3)?,
        revision: row.get(4)?,
    })
}

fn read_item(row: &Row<'_>) -> rusqlite::Result<ListItem> {
    Ok(ListItem {
        id: ItemId(row.get(0)?),
        text: row.get(1)?,
        completed: row.get(2)?,
        ha_uid: row.get(3)?,
    })
}

fn get_list(db: &Connection, id: ListId, owner: Option<&str>) -> Result<Option<List>> {
    Ok(db
        .query_row(
            &format!("SELECT {LIST_COLUMNS} FROM lists WHERE id = ?1 AND owner IS ?2"),
            params![id.0, owner],
            read_list,
        )
        .optional()?)
}

fn get_items(db: &Connection, id: ListId) -> Result<Vec<ListItem>> {
    query_all(
        db,
        &format!("SELECT {ITEM_COLUMNS} FROM list_items WHERE list_id = ?1 ORDER BY id"),
        [id.0],
        read_item,
    )
}

fn require_list(db: &Connection, id: ListId, owner: Option<&str>) -> Result<List> {
    get_list(db, id, owner)?.ok_or(Error::NotFound("list"))
}

fn update_revision(db: &Connection, id: ListId, changes: usize) -> Result<()> {
    if changes > 0 {
        db.execute(
            "UPDATE lists SET revision = revision + 1 WHERE id = ?1",
            [id.0],
        )?;
    }
    Ok(())
}

impl<C: Clock> Household<C> {
    pub fn create_list(
        &mut self,
        name: &str,
        owner: Option<&str>,
        ha_entity: Option<&str>,
    ) -> Result<List> {
        required(name, "list name is required")?;
        if let Some(owner) = owner {
            required(owner, "personal list owner is required")?;
        }
        if let Some(entity) = ha_entity
            && (owner.is_some()
                || !entity.strip_prefix("todo.").is_some_and(|name| {
                    !name.is_empty()
                        && name.bytes().all(|byte| {
                            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
                        })
                }))
        {
            return Err(Error::InvalidInput(
                "a valid todo entity requires a shared list",
            ));
        }
        self.transaction(|db| {
            Ok(db.query_row(
                &format!(
                    "INSERT INTO lists(name, name_key, owner, ha_entity)
                      VALUES (?1, ?2, ?3, ?4) RETURNING {LIST_COLUMNS}"
                ),
                params![name.trim(), name_key(name), owner, ha_entity],
                read_list,
            )?)
        })
    }

    pub fn lists(&self, owner: Option<&str>) -> Result<Vec<List>> {
        query_all(
            self.db(),
            &format!("SELECT {LIST_COLUMNS} FROM lists WHERE owner IS ?1 ORDER BY id"),
            [owner],
            read_list,
        )
    }

    pub fn list_snapshot(
        &mut self,
        id: ListId,
        owner: Option<&str>,
    ) -> Result<Option<ListSnapshot>> {
        let db = self.db().unchecked_transaction()?;
        let snapshot = get_list(&db, id, owner)?
            .map(|list| get_items(&db, id).map(|items| ListSnapshot { list, items }))
            .transpose()?;
        db.commit()?;
        Ok(snapshot)
    }

    pub fn rename_list(
        &mut self,
        id: ListId,
        owner: Option<&str>,
        name: &str,
    ) -> Result<Option<List>> {
        required(name, "list name is required")?;
        self.transaction(|db| {
            Ok(db
                .query_row(
                    &format!(
                        "UPDATE lists SET name = ?3, name_key = ?4, revision = revision + 1
                      WHERE id = ?1 AND owner IS ?2 RETURNING {LIST_COLUMNS}"
                    ),
                    params![id.0, owner, name.trim(), name_key(name)],
                    read_list,
                )
                .optional()?)
        })
    }

    pub fn delete_list(&mut self, id: ListId, owner: Option<&str>) -> Result<bool> {
        Ok(self.db().execute(
            "DELETE FROM lists WHERE id = ?1 AND owner IS ?2",
            params![id.0, owner],
        )? == 1)
    }

    pub fn add_list_item(
        &mut self,
        id: ListId,
        owner: Option<&str>,
        text: &str,
    ) -> Result<ListItem> {
        required(text, "list item text is required")?;
        self.transaction(|db| {
            require_list(db, id, owner)?;
            let item = db.query_row(
                &format!(
                    "INSERT INTO list_items(list_id, text) VALUES (?1, ?2) RETURNING {ITEM_COLUMNS}"
                ),
                params![id.0, text.trim()],
                read_item,
            )?;
            update_revision(db, id, 1)?;
            Ok(item)
        })
    }

    pub fn update_list_item(
        &mut self,
        list: ListId,
        owner: Option<&str>,
        item: ItemId,
        text: &str,
        completed: bool,
    ) -> Result<bool> {
        required(text, "list item text is required")?;
        self.transaction(|db| {
            require_list(db, list, owner)?;
            let changed = db.execute(
                "UPDATE list_items SET text = ?3, completed = ?4 WHERE list_id = ?1 AND id = ?2",
                params![list.0, item.0, text.trim(), completed],
            )?;
            update_revision(db, list, changed)?;
            Ok(changed == 1)
        })
    }

    pub fn delete_list_item(
        &mut self,
        list: ListId,
        owner: Option<&str>,
        item: ItemId,
    ) -> Result<bool> {
        self.transaction(|db| {
            require_list(db, list, owner)?;
            let changed = db.execute(
                "DELETE FROM list_items WHERE list_id = ?1 AND id = ?2",
                params![list.0, item.0],
            )?;
            update_revision(db, list, changed)?;
            Ok(changed == 1)
        })
    }

    pub fn clear_list(&mut self, list: ListId, owner: Option<&str>) -> Result<usize> {
        self.transaction(|db| {
            require_list(db, list, owner)?;
            let changed = db.execute("DELETE FROM list_items WHERE list_id = ?1", [list.0])?;
            update_revision(db, list, changed)?;
            Ok(changed)
        })
    }

    /// Apply HA changes only to the exact revision sent to the adapter.
    pub fn apply_todo_changes(
        &mut self,
        id: ListId,
        revision: i64,
        changes: &[TodoChange],
    ) -> Result<ListSnapshot> {
        self.transaction(|db| {
            let mut list = require_list(db, id, None)?;
            if list.ha_entity.is_none() || list.revision != revision {
                return Err(Error::Conflict(
                    "todo list configuration or revision changed",
                ));
            }
            for change in changes {
                apply_todo_change(db, id, change)?;
            }
            update_revision(db, id, changes.len())?;
            if !changes.is_empty() {
                list = require_list(db, id, None)?;
            }
            Ok(ListSnapshot {
                list,
                items: get_items(db, id)?,
            })
        })
    }
}

fn apply_todo_change(db: &Connection, list: ListId, change: &TodoChange) -> Result<()> {
    let affected = match change {
        TodoChange::Add {
            uid,
            text,
            completed,
        } => {
            validate_remote_item(uid, text)?;
            db.execute(
                "INSERT INTO list_items(list_id, ha_uid, text, completed) VALUES (?1, ?2, ?3, ?4)",
                params![list.0, uid, text.trim(), completed],
            )?
        }
        TodoChange::Update {
            item,
            uid,
            text,
            completed,
        } => {
            validate_remote_item(uid, text)?;
            db.execute("UPDATE list_items SET ha_uid = ?3, text = ?4, completed = ?5 WHERE list_id = ?1 AND id = ?2",
                params![list.0, item.0, uid, text.trim(), completed])?
        }
        TodoChange::Remove(item) => db.execute(
            "DELETE FROM list_items WHERE list_id = ?1 AND id = ?2",
            params![list.0, item.0],
        )?,
    };
    if affected != 1 {
        return Err(Error::NotFound("todo item"));
    }
    Ok(())
}

fn validate_remote_item(uid: &str, text: &str) -> Result<()> {
    required(uid, "HA todo item requires a UID")?;
    required(text, "HA todo item text is required")
}

impl<C: Clock + 'static> HouseholdWorker<C> {
    /// Reconcile by calling Home Assistant outside the SQLite worker.
    pub async fn sync_todo<S: TodoSync>(
        &self,
        id: ListId,
        adapter: &S,
    ) -> std::result::Result<ListSnapshot, TodoSyncError<S::Error>> {
        let snapshot = self
            .call(move |household| household.list_snapshot(id, None))
            .await?
            .ok_or(Error::NotFound("shared list"))?;
        let entity = snapshot
            .list
            .ha_entity
            .as_deref()
            .ok_or(Error::Conflict("list has no HA todo entity"))?;
        let changes = adapter
            .reconcile(entity, &snapshot)
            .await
            .map_err(TodoSyncError::Adapter)?;
        let revision = snapshot.list.revision;
        Ok(self
            .call(move |household| household.apply_todo_changes(id, revision, &changes))
            .await?)
    }
}
