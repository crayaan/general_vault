# State Hoisting

A pattern for managing state in Jetpack Compose UI components.

## Core Concept
- **Definition**
  - Lifting state up to caller
  - Making components stateless
  - Enabling reusability
- **Benefits**
  - Single source of truth
  - Testability
  - Predictable behavior

## Implementation Pattern
- **Stateful vs Stateless**
  - Internal state (convenience)
  - Hoisted state (control)
  - Hybrid approaches
- **Parameter Structure**
  - Value parameter
  - On-change callback
  - Example: `value: T, onValueChange: (T) -> Unit`

## Example Implementation
```kotlin
// Stateless TextField
@Composable
fun CustomTextField(
    value: String, 
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    // other parameters
) {
    TextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier
    )
}

// Usage with hoisted state
@Composable
fun Screen() {
    var text by remember { mutableStateOf("") }
    
    CustomTextField(
        value = text,
        onValueChange = { text = it }
    )
}
```

## Common Patterns
- **Remember Helper**
  - Creating convenience wrappers
  - Providing both stateful and stateless versions
  - Example with default implementation
```kotlin
// Stateful wrapper around stateless component
@Composable
fun StatefulCustomTextField(
    initialValue: String = "",
    modifier: Modifier = Modifier
) {
    var text by remember { mutableStateOf(initialValue) }
    
    CustomTextField(
        value = text,
        onValueChange = { text = it },
        modifier = modifier
    )
}
```

## Related Topics
- [[State Management]] - Overall state handling approaches
- [[Component API Design]] - How state impacts API design
- [[UI Component Design]] - Creating effective components
- [[Remember and Mutable State]] - State preservation

## Next Steps
→ Learn about [[Component API Design]] for building robust components

---
Tags: #jetpack-compose #state-management #state-hoisting #patterns 