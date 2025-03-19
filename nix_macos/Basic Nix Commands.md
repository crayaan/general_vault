# Basic Nix Commands

Nix provides a powerful set of commands for managing packages, development environments, and system configurations. This guide covers essential Nix commands to get you started on macOS.

## Overview

Nix has evolved to include two command interfaces:
1. **Legacy Commands**: Older style commands like `nix-env`, `nix-shell`, etc.
2. **New-style Commands**: Modern unified `nix` command with subcommands (preferred)

This guide will focus primarily on the new-style commands while mentioning legacy equivalents when relevant. The new command structure is more consistent and includes enhanced features like flakes support.

## Package Management

### Installing Packages

#### Temporarily Run a Package

Run a package without installing it permanently:

```bash
# New style
nix run nixpkgs#package-name

# Example
nix run nixpkgs#ripgrep
```

#### Install to User Profile

Install a package to your user environment:

```bash
# New style
nix profile install nixpkgs#package-name

# Legacy equivalent
nix-env -iA nixpkgs.package-name
```

#### Install Multiple Packages

```bash
nix profile install nixpkgs#package1 nixpkgs#package2
```

### Searching for Packages

Find available packages in nixpkgs:

```bash
# Search by name
nix search nixpkgs package-name

# Search with regex
nix search nixpkgs '.*regex.*'
```

### Listing Installed Packages

View packages in your profile:

```bash
# New style
nix profile list

# Legacy equivalent
nix-env -q
```

### Removing Packages

Remove a package from your profile:

```bash
# New style - first list to get the index number
nix profile list
# Then remove by index
nix profile remove INDEX

# Legacy equivalent
nix-env -e package-name
```

### Updating Packages

Update all packages in your profile:

```bash
# New style
nix profile upgrade '.*'

# Legacy equivalent
nix-env -u
```

## Development Environments

### Create a Temporary Shell Environment

Launch a shell with specific packages available:

```bash
# With specific packages
nix shell nixpkgs#package1 nixpkgs#package2

# Legacy equivalent
nix-shell -p package1 package2
```

### Development Shell from a Flake

If a project has a `flake.nix` file:

```bash
# Enter the development environment
nix develop

# Run a command in the environment without entering it
nix develop --command command-to-run
```

### Ad-hoc Development Environment

Create a quick environment for specific languages:

```bash
# Python environment
nix shell nixpkgs#python3 nixpkgs#python3Packages.numpy

# Node.js environment
nix shell nixpkgs#nodejs nixpkgs#yarn
```

## Working with Flakes

### Initialize a New Flake

Create a new flake in the current directory:

```bash
# Create a basic flake.nix file
mkdir -p my-project
cd my-project
nix flake init
```

### Update Flake Inputs

Update the dependencies in a flake:

```bash
nix flake update
```

### Show Flake Information

Display information about a flake:

```bash
nix flake show
```

### Build a Flake Output

Build a specific output from a flake:

```bash
nix build .#output-name
```

## Garbage Collection and Maintenance

### Run Garbage Collection

Clean up unused Nix store paths:

```bash
# Basic garbage collection
nix store gc

# Legacy equivalent
nix-collect-garbage
```

### Optimize Store

Optimize the Nix store to save space:

```bash
nix store optimise
```

### Check Store Health

Verify integrity of the Nix store:

```bash
nix store verify --all
```

## System Information and Configuration

### Show Nix Configuration

Display current Nix configuration:

```bash
nix show-config
```

### Check Nix Version

Display Nix version information:

```bash
nix --version
```

## macOS-Specific Commands

### Manage Nix Darwin (if installed)

Rebuild your macOS system configuration:

```bash
# With a flake
darwin-rebuild switch --flake .

# Without a flake (legacy)
darwin-rebuild switch
```

### Check Native vs Intel Packages (Apple Silicon)

On Apple Silicon Macs, you can specify architecture:

```bash
# Run native ARM64 version
nix run nixpkgs#legacyPackages.aarch64-darwin.package-name

# Run Intel version (requires Rosetta 2)
nix run nixpkgs#legacyPackages.x86_64-darwin.package-name
```

## Troubleshooting and Information

### Show Build Logs

View detailed build logs for a derivation:

```bash
nix log /nix/store/HASH-name
```

### Get Package Information

Display detailed information about a package:

```bash
nix show-derivation /nix/store/HASH-name
```

### Show Path Information

Get details about a Nix store path:

```bash
nix path-info /nix/store/HASH-name
```

## Useful Command Patterns

### Create a Quick Environment for a Task

```bash
# Create a temporary Python data science environment
nix shell nixpkgs#python3 nixpkgs#python3Packages.pandas nixpkgs#python3Packages.matplotlib
```

### View Documentation

```bash
# View man pages for a command without installing it
nix run nixpkgs#man -c man some-command
```

### Install macOS GUI Applications

```bash
# Install a GUI application (when available in Nix)
nix profile install nixpkgs#vlc
```

## Common Command Flags

- `--verbose` or `-v`: Increase verbosity
- `--quiet` or `-q`: Decrease verbosity
- `--help`: Show help for a command
- `--option sandbox false`: Disable sandboxing (useful on macOS for certain builds)
- `--impure`: Allow impure operations (for commands that need network access)

## Relation to Other Concepts

- **[[What is Nix]]**: Understanding the foundational concepts
- **[[Nix Flakes Introduction]]**: More detailed exploration of flakes
- **[[Working with Nix Packages]]**: In-depth package management
- **[[Nix Shells for Development]]**: Comprehensive guide to development environments

## Next Steps
→ Learn about [[Nix Flakes Introduction]] for modern Nix workflows
→ Explore [[Nix Shells for Development]] for project-specific environments
→ Get familiar with [[Working with Nix Packages]] for more package management options

---
Tags: #nix #commands #cli #package-management #macos 