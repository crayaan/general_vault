# Jetpack Compose Basics

Fundamental concepts and components for building UIs with Jetpack Compose.

## Composable Functions
- **Definition**
  - Functions annotated with `@Composable`
  - Declarative UI building blocks
  - Can emit UI elements or other composables
- **Characteristics**
  - Do not return values (typically)
  - Can be called from other composables
  - Execute in a composition context
- **Example**
```kotlin
@Composable
fun Greeting(name: String) {
    Text(text = "Hello $name!")
}
```

## Modifiers
- **Purpose**
  - Styling and layout configuration
  - Chained to customize components
  - Applied in specific order
- **Common Modifiers**
  - `Modifier.size()` - Sets dimensions
  - `Modifier.padding()` - Adds space around element
  - `Modifier.background()` - Sets background appearance
  - `Modifier.clickable()` - Adds click behavior
- **Chaining**
```kotlin
Modifier
    .padding(16.dp)
    .size(200.dp)
    .background(Color.Blue)
    .clickable { /* action */ }
```

## Layout Basics
- **Row and Column**
  - `Row` arranges items horizontally
  - `Column` arranges items vertically
  - Alignment and arrangement options
- **Box**
  - Overlay elements on top of each other
  - Z-order based on composition order
- **Scaffold**
  - Material Design layout structure
  - TopBar, BottomBar, FAB, Drawer support

## Related Topics
- [[State Management]] - How to handle UI state
- [[Advanced Compose Concepts]] - More complex concepts
- [[Layout Fundamentals]] - Deeper layout understanding

## Next Steps
→ Explore [[State Management]] to understand how to manage UI state in Compose

---
Tags: #jetpack-compose #basics #composable #modifiers 