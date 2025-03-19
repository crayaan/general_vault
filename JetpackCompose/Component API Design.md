# Component API Design

Guidelines for creating effective, consistent, and user-friendly APIs for Jetpack Compose components.

## API Structure
- **Parameter Order**
  - Required parameters first
  - State and callbacks next
  - Optional parameters last
  - Modifier always after required parameters
- **Naming Conventions**
  - Descriptive function names
  - Consistent parameter naming
  - Contextual prefixing when needed

## State Handling
- **State Hoisting**
  - Provide value + callback patterns
  - Allow state to be controlled externally
  - See [[State Hoisting]] for details
- **Default Implementations**
  - Provide both stateful and stateless versions
  - Internal state management for convenience
  - External state control for complex use cases

## Modifier Usage
- **Always Accept Modifiers**
  - Every component should accept a modifier parameter
  - Use `Modifier = Modifier` as default
  - Pass modifiers to the outermost composable
- **Modifier Application**
  - Apply in the order specified by the caller
  - Combine with internal modifiers carefully
  - See [[Modifier Best Practices]]

## Content Slots
- **Content Lambdas**
  - Use content lambdas for customizable sections
  - Provide sensible defaults when possible
  - Use multiple content slots for complex components
- **Example**
```kotlin
@Composable
fun CustomCard(
    title: String,
    modifier: Modifier = Modifier,
    titleContent: @Composable () -> Unit = { Text(title) },
    content: @Composable () -> Unit
) {
    Card(modifier = modifier) {
        Column {
            Box(Modifier.padding(16.dp)) {
                titleContent()
            }
            Divider()
            Box(Modifier.padding(16.dp)) {
                content()
            }
        }
    }
}
```

## Error Handling
- **Input Validation**
  - Check input parameters when appropriate
  - Fail fast with clear error messages
  - Include parameter names in error messages
- **Graceful Degradation**
  - Handle edge cases gracefully
  - Provide fallback UI for error states
  - Document limitations and requirements

## Related Topics
- [[UI Component Design]] - Overall component design
- [[State Hoisting]] - State management patterns
- [[Modifier Best Practices]] - Handling modifiers
- [[Library Development]] - Creating a component library

## Next Steps
→ Apply these principles to your components in [[UI Component Design]]

---
Tags: #jetpack-compose #api-design #components #best-practices 