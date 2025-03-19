# Nix Flakes Introduction

Nix Flakes are a modern approach to managing Nix packages and configurations with improved reproducibility, composability, and usability. This guide introduces flakes with a practical focus for macOS users.

## What Are Nix Flakes?

Flakes are a Nix feature that provides a standardized way to:
- Package Nix expressions with explicit dependencies
- Make projects easier to share and compose
- Improve reproducibility with lockfiles (similar to `package-lock.json` or `Cargo.lock`)
- Simplify the management of inputs and outputs

Flakes address many of the challenges with traditional Nix, such as ambiguous channels, impure evaluation, and complex dependency management.

## Key Benefits of Flakes

- **Reproducibility**: Locked dependencies ensure consistent builds across environments
- **Discoverability**: Standardized structure makes flake outputs easy to discover
- **Composability**: Flakes can easily reference and build upon each other
- **Portability**: Clear dependency specification makes sharing projects simpler
- **Caching**: Better caching for faster builds and development

## Enabling Flakes on macOS

Flakes are considered experimental but are widely used. If you installed Nix using the Determinate Systems installer, flakes are enabled by default. Otherwise:

1. Create or edit `/etc/nix/nix.conf` (or `~/.config/nix/nix.conf`):
   ```
   experimental-features = nix-command flakes
   ```

2. If using `nix-darwin`, add to your configuration:
   ```nix
   nix.settings.experimental-features = [ "nix-command" "flakes" ];
   ```

## Anatomy of a Flake

A flake is defined by two key files:

1. **flake.nix**: The flake definition with inputs and outputs
2. **flake.lock**: A lockfile containing precise versions of dependencies

### Basic flake.nix Structure

```nix
{
  description = "A simple flake";

  inputs = {
    # Dependencies
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        # Outputs defined here
        packages.default = pkgs.hello;
        
        devShells.default = pkgs.mkShell {
          buildInputs = [ pkgs.hello ];
        };
      }
    );
}
```

## Flake Inputs

Inputs define dependencies that your flake relies on:

```nix
inputs = {
  nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  
  # Specific version (using a git rev)
  nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-23.11";
  
  # Local path
  my-local-flake.url = "path:/path/to/local/flake";
  
  # Input with input (transitive dependency)
  utils.url = "github:numtide/flake-utils";
  utils.inputs.nixpkgs.follows = "nixpkgs";
};
```

### Common Input Sources

- **GitHub**: `github:owner/repo/ref`
- **GitLab**: `gitlab:owner/repo/ref`
- **Local path**: `path:/absolute/path` or `path:./relative/path`
- **URL**: `https://example.com/flake.tar.gz`
- **Flake registry**: `nixpkgs` or `flake-utils`

## Flake Outputs

Outputs are what your flake provides to others. Common outputs include:

```nix
outputs = { self, nixpkgs, ... }: {
  # Packages
  packages.x86_64-darwin.default = ...;
  packages.aarch64-darwin.default = ...;
  
  # Development shells
  devShells.x86_64-darwin.default = ...;
  
  # NixOS or nix-darwin modules
  darwinModules.myModule = ...;
  
  # Overlays
  overlays.default = ...;
  
  # Applications
  apps.x86_64-darwin.default = ...;
};
```

## Working with Flakes on macOS

### Create a New Flake

Initialize a new flake in the current directory:

```bash
mkdir my-project
cd my-project
nix flake init
```

### Using Templates

Create a new flake from a template:

```bash
nix flake init -t github:NixOS/templates#rust
```

View available templates:

```bash
nix flake show github:NixOS/templates
```

### Managing the Lock File

Update all inputs to their latest versions:

```bash
nix flake update
```

Update a specific input:

```bash
nix flake update --update-input nixpkgs
```

### Development Shell with Flakes

Enter a development environment:

```bash
nix develop
```

Run a command in the development environment:

```bash
nix develop --command zsh
```

### Building and Running

Build the default package:

```bash
nix build
```

Build a specific output:

```bash
nix build .#packages.aarch64-darwin.myapp
```

Run a flake app:

```bash
nix run
```

## Practical Flake Examples for macOS

### Simple Development Environment

```nix
{
  description = "Development environment for Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            python3Packages.numpy
            python3Packages.pandas
          ];
        };
      }
    );
}
```

### macOS Application with nix-darwin

```nix
{
  description = "My macOS configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    darwin.url = "github:lnl7/nix-darwin";
    darwin.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, darwin }: {
    darwinConfigurations.mymac = darwin.lib.darwinSystem {
      system = "aarch64-darwin";
      modules = [
        {
          nixpkgs.config.allowUnfree = true;
          environment.systemPackages = with nixpkgs.legacyPackages.aarch64-darwin; [
            git
            vim
          ];
        }
      ];
    };
  };
}
```

### Cross-Platform Package with Apple Silicon Support

```nix
{
  description = "Cross-platform project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        isAppleSilicon = system == "aarch64-darwin";
        extraBuildInputs = if isAppleSilicon 
          then [ pkgs.darwin.apple_sdk.frameworks.CoreServices ] 
          else [];
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "my-package";
          src = self;
          buildInputs = [ pkgs.hello ] ++ extraBuildInputs;
          # ... rest of the derivation
        };
      }
    );
}
```

## Best Practices for Flakes

1. **Pinned Dependencies**: Use the lockfile for reproducibility
2. **Clear Structure**: Follow standard flake conventions
3. **Cross-Platform Support**: Consider multiple systems (especially aarch64-darwin for Apple Silicon)
4. **Minimize Inputs**: Only include what you need
5. **Descriptive Names**: Use clear names for inputs and outputs
6. **Document Outputs**: Make it clear what your flake provides
7. **Composable Modules**: Design components to be easily reused

## Troubleshooting Flakes on macOS

### Common Issues

- **"error: experimental feature 'flakes' is disabled"**: Enable experimental features in the Nix configuration
- **Building fails with missing libraries**: Ensure macOS SDK paths are correctly configured
- **Differences between Intel and ARM builds**: Use system-specific configuration 
- **Error with macOS frameworks**: Add required frameworks to `buildInputs`

### Debugging Flake Evaluation

View flake metadata:
```bash
nix flake metadata
```

Show flake outputs:
```bash
nix flake show
```

Check flake inputs:
```bash
nix flake info
```

## Relation to Other Concepts

- **[[What is Nix]]**: Foundation for understanding Nix ecosystem
- **[[Basic Nix Commands]]**: Commands used with flakes
- **[[Working with Nix Packages]]**: Package management in Nix
- **[[Installing Nix on macOS]]**: Setting up Nix with flakes support

## Next Steps

→ Create a [[Nix Development Workflows]] for your projects
→ Learn about [[Nix Darwin Configuration]] for full macOS system management
→ Explore [[Managing Nix Environments]] for complex development needs

---
Tags: #nix #flakes #package-management #reproducible-builds #macos 