# State Management in Compose

Understanding how to manage and update UI state in a declarative framework.

## State in Compose
- **Definition**
  - Any value that can change over time
  - Triggers recomposition when changed
  - Source of truth for UI elements
- **Types of State**
  - UI State (text input, selection, etc.)
  - App State (user data, preferences, etc.)
  - Navigation State (current screen, etc.)

## Remember and MutableState
- **`remember`**
  - Preserves state across recompositions
  - Lost when composable leaves composition
  - Local to a specific composable instance
- **`mutableStateOf`**
  - Creates observable state objects
  - Changes trigger recomposition
  - Usually wrapped with `remember`
- **Example**
```kotlin
@Composable
fun Counter() {
    var count by remember { mutableStateOf(0) }
    
    Button(onClick = { count++ }) {
        Text("Count: $count")
    }
}
```

## State Hoisting
- **Concept**
  - Moving state up to caller
  - Making components stateless
  - Enabling reusability and testability
- **Implementation Pattern**
  - Value + onChange callback
  - Single source of truth
  - Enhanced component flexibility
- **Read full details in [[State Hoisting]]**

## State Holders
- **ViewModel Integration**
  - Separation of UI and business logic
  - Surviving configuration changes
  - Managing complex state
- **Composable State Holders**
  - `remember { MyStateHolder() }`
  - Encapsulating related state
  - Organizing business logic

## Related Topics
- [[Remember and Mutable State]] - Detailed remember usage
- [[State Hoisting]] - Pattern for lifting state
- [[Recomposition]] - How Compose updates the UI
- [[Jetpack Compose Basics]] - Fundamental concepts

## Next Steps
→ Learn specifically about [[State Hoisting]] for creating reusable components

---
Tags: #jetpack-compose #state-management #remember #mutablestate 