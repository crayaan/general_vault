# Installing Nix on macOS

Installing Nix on macOS requires a few specific steps to ensure proper integration with Apple's security features and filesystem structure. This guide walks through the installation process with special attention to Apple Silicon considerations.

## Overview

Nix installation on macOS differs from Linux installations due to macOS-specific constraints. Recent versions of macOS include security features like System Integrity Protection (SIP) and the read-only system volume, which require special handling. Modern Nix installers address these challenges by creating a dedicated APFS volume for the Nix store.

The installation process is straightforward with the right tools, and in most cases can be completed in under 10 minutes.

## Prerequisites

Before installing Nix, ensure your system meets these requirements:

- **macOS Version**: macOS 10.15 (Catalina) or newer is recommended; this guide focuses on macOS Ventura/Sonoma
- **Disk Space**: At least 10GB of free disk space
- **Command Line Tools**: Xcode Command Line Tools installed
- **Admin Access**: Administrator privileges on your Mac

### Installing Xcode Command Line Tools

If you haven't already installed the Xcode Command Line Tools, run:

```bash
xcode-select --install
```

This will prompt you to install the necessary developer tools without requiring the full Xcode application.

## Installation Options

There are three primary methods to install Nix on macOS:

1. **Determinate Systems Installer** (Recommended): A modern installer with enhanced macOS integration
2. **Official Nix Installer**: The standard installer from the Nix project
3. **Determinate Systems GUI Installer**: A graphical installation experience

This guide will focus on the Determinate Systems installer as it offers several benefits for macOS users:

- Designed to survive macOS updates
- Better integration with macOS security features
- Enables Nix Flakes by default
- More user-friendly installation process

## Installation Steps

### Method 1: Determinate Systems Installer (Recommended)

1. **Open Terminal**: Launch the Terminal application from Applications/Utilities or Spotlight

2. **Run the Installer**: Execute the installation command:

```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

3. **Review and Confirm**: The installer will explain what changes it will make to your system. Review these and confirm to proceed.

4. **Provide Authentication**: Enter your administrator password when prompted

5. **Wait for Completion**: The installer will:
   - Create a dedicated APFS volume for the Nix store
   - Configure the Nix daemon
   - Set up necessary system files
   - Configure your shell environment

6. **Restart Your Shell**: Once the installation is complete, either restart your terminal or run:

```bash
source ~/.zshrc  # Or the appropriate config file for your shell
```

7. **Verify Installation**: Test that Nix is working properly:

```bash
nix --version
nix run nixpkgs#hello
```

If you see the version information and "Hello, world!" output, Nix is installed correctly.

### Method 2: Official Nix Installer

If you prefer to use the official installer from the Nix project:

```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

This installer offers fewer macOS-specific optimizations but works well for standard installations.

### Method 3: Determinate Systems GUI Installer (macOS App)

For those who prefer a graphical interface:

1. Download the installer from [Determinate Systems GitHub](https://github.com/DeterminateSystems/nix-installer-app/releases)
2. Open the downloaded `.dmg` file
3. Drag the Nix Installer to your Applications folder
4. Run the application and follow the on-screen instructions

This method is particularly useful for enterprise environments using mobile device management (MDM) solutions.

## Apple Silicon Specific Considerations

If you're using a Mac with Apple Silicon (M1, M2, or M3 processors), there are additional considerations:

### Rosetta 2 for Intel Compatibility

To allow Nix to work with x86_64 (Intel) packages on Apple Silicon:

1. **Install Rosetta 2**:

```bash
softwareupdate --install-rosetta --agree-to-license
```

2. **Enable Multi-Architecture Support** (can be added to your Nix configuration later):

```
nix.extraOptions = ''
  extra-platforms = x86_64-darwin aarch64-darwin
'';
```

### Native vs. Intel Packages

By default, Nix on Apple Silicon will use native ARM64 packages (labeled as `aarch64-darwin`) when available. Some packages may still only be available as Intel packages (labeled as `x86_64-darwin`), which will run through Rosetta 2 translation.

To explicitly specify architecture when running a package:

```bash
# For native ARM64
nix run "nixpkgs#legacyPackages.aarch64-darwin.hello"

# For Intel via Rosetta 2
nix run "nixpkgs#legacyPackages.x86_64-darwin.hello"
```

## Post-Installation Configuration

After installing Nix, there are several recommended steps to optimize your setup:

### 1. Enable Flakes (if not already enabled)

Flakes provide a more structured way to manage Nix packages and configurations. If your installer didn't enable them automatically:

Add to `~/.config/nix/nix.conf`:

```
experimental-features = nix-command flakes
```

### 2. Configure Shell Integration

Ensure your shell is properly configured by adding to your shell configuration file (e.g., `~/.zshrc`):

```bash
# For multi-user installations
if [ -e '/nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh' ]; then
  . '/nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh'
fi
```

### 3. Set Default Channels

If you're not using flakes exclusively, configure a default channel:

```bash
nix-channel --add https://nixos.org/channels/nixpkgs-unstable nixpkgs
nix-channel --update
```

## Troubleshooting Common Issues

### Installation Fails with Permission Errors

**Issue**: The installer fails with permission-related errors.

**Solution**:
1. Ensure you have administrator privileges
2. Check that SIP is not interfering with the installation
3. Try running the installer with `sudo` (for some installation methods)

### Shell Commands Not Found After Installation

**Issue**: Nix commands are not available after installation.

**Solution**:
1. Ensure shell integration is properly configured
2. Restart your terminal or source your shell configuration file
3. Check that the Nix daemon is running: `sudo launchctl list | grep nix`

### Problems with Apple Silicon Architecture

**Issue**: Packages fail with architecture-related errors.

**Solution**:
1. Ensure Rosetta 2 is installed
2. Explicitly specify the architecture when running commands
3. Check if the package has native ARM64 support

## Uninstalling Nix

If you need to uninstall Nix:

### For Determinate Systems Installer:

```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- uninstall
```

### For Official Installer:

```bash
sudo rm -rf /nix
# And remove references in shell config files
```

## Relation to Other Concepts

- **[[What is Nix]]**: Understand the fundamental concepts behind Nix
- **[[Basic Nix Commands]]**: Learn essential commands to start using Nix
- **[[Setting Up nix-darwin]]**: Configure macOS system settings declaratively
- **[[Apple Silicon Considerations]]**: More details on using Nix with M1/M2/M3 Macs

## Next Steps
→ Explore [[Basic Nix Commands]] to get started with Nix
→ Set up [[Nix Shells for Development]] for project-specific environments
→ Learn about [[Nix Flakes Introduction]] for modern Nix workflows

---
Tags: #nix #macos #installation #apple-silicon #package-manager 