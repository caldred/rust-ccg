//! Card system: definitions, instances, and registry.
//!
//! ## Key Types
//!
//! - `CardId`: Identifier for card definitions
//! - `CardTypeId`: Opaque type identifier (games define types)
//! - `CardDefinition`: Static card data with generic attributes
//! - `CardInstance`: Runtime card state (zone, owner, counters)
//! - `CardRegistry`: Card definition lookup
//!
//! ## Neutral Cards
//!
//! Cards can have `owner: None` for deckbuilders with shared markets,
//! neutral obstacles, or game-controlled entities.

pub mod attributes;
pub mod definition;
pub mod instance;
pub mod registry;

pub use attributes::{AttributeKey, AttributeValue, Attributes};
pub use definition::{CardDefinition, CardId, CardTypeId};
pub use instance::CardInstance;
pub use registry::CardRegistry;
