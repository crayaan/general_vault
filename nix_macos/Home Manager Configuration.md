# Home Manager Configuration

Home Manager is a powerful tool for managing user-specific packages and dotfiles using Nix. This guide focuses on setting up and configuring Home Manager on macOS, with an emphasis on practical examples and workflows.

## What is Home Manager?

Home Manager brings the declarative power of Nix to user environment management:

- **Manage user packages** independently from system packages
- **Configure dotfiles** (.zshrc, .gitconfig, etc.) declaratively
- **Set up user applications** consistently across systems
- **Version control** your entire user environment
- **Safely roll back** to previous configurations

While nix-darwin manages system-wide settings, Home Manager focuses on user-specific configuration.

## Installation

### Prerequisites

- Nix package manager installed (see [[Installing Nix on macOS]])
- Recommended: Flakes enabled (see [[Nix Flakes Introduction]])

### Installation Methods

#### Standalone Installation with Flakes (Recommended)

1. Create a directory for your configuration:
   ```bash
   mkdir -p ~/.config/home-manager
   cd ~/.config/home-manager
   ```

2. Create a basic `flake.nix` file:
   ```nix
   {
     description = "Home Manager configuration";
   
     inputs = {
       nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
       home-manager = {
         url = "github:nix-community/home-manager";
         inputs.nixpkgs.follows = "nixpkgs";
       };
     };
   
     outputs = { nixpkgs, home-manager, ... }: {
       homeConfigurations."yourusername" = home-manager.lib.homeManagerConfiguration {
         pkgs = nixpkgs.legacyPackages.aarch64-darwin; # or x86_64-darwin for Intel
         modules = [ ./home.nix ];
       };
     };
   }
   ```

3. Create a basic `home.nix` file:
   ```nix
   { config, pkgs, ... }: {
     # Home Manager needs a bit of information about you and the paths it should manage
     home.username = "yourusername";
     home.homeDirectory = "/Users/yourusername";
   
     # Packages to install
     home.packages = with pkgs; [
       ripgrep
       fd
       bat
       jq
     ];
     
     # Enable Home Manager
     programs.home-manager.enable = true;
     
     # Create XDG directories
     xdg.enable = true;
     
     # This value determines the Home Manager release that your
     # configuration is compatible with
     home.stateVersion = "23.11";
   }
   ```

4. Build and activate your configuration:
   ```bash
   nix run home-manager/master -- switch --flake .
   ```

#### Integration with nix-darwin

If you're using nix-darwin (see [[Nix Darwin Configuration]]), you can integrate Home Manager:

1. Add Home Manager to your `flake.nix` inputs:
   ```nix
   inputs = {
     nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
     darwin.url = "github:lnl7/nix-darwin";
     darwin.inputs.nixpkgs.follows = "nixpkgs";
     home-manager.url = "github:nix-community/home-manager";
     home-manager.inputs.nixpkgs.follows = "nixpkgs";
   };
   ```

2. Add Home Manager as a module in your `darwinConfigurations`:
   ```nix
   darwinConfigurations."your-hostname" = darwin.lib.darwinSystem {
     system = "aarch64-darwin";
     modules = [
       ./configuration.nix
       home-manager.darwinModules.home-manager
       {
         home-manager.useGlobalPkgs = true;
         home-manager.useUserPackages = true;
         home-manager.users.yourusername = import ./home.nix;
       }
     ];
   };
   ```

3. Build and activate both your nix-darwin and Home Manager configurations:
   ```bash
   darwin-rebuild switch --flake .
   ```

## Core Configuration Areas

### Package Management

Install user-specific packages:

```nix
home.packages = with pkgs; [
  # Command line tools
  ripgrep
  fd
  bat
  jq
  httpie
  
  # Development tools
  yarn
  nodejs
  rustup
  
  # GUI applications (available in nixpkgs)
  vscode
  obsidian
];
```

### Shell Configuration

Configure your preferred shell:

```nix
programs.zsh = {
  enable = true;
  enableAutosuggestions = true;
  enableCompletion = true;
  
  initExtra = ''
    # Custom shell initialization
    bindkey -e  # Emacs key bindings
  '';
  
  shellAliases = {
    ll = "ls -la";
    g = "git";
    vim = "nvim";
  };
  
  oh-my-zsh = {
    enable = true;
    plugins = [ "git" "macos" "docker" ];
    theme = "robbyrussell";
  };
  
  plugins = [
    {
      name = "zsh-syntax-highlighting";
      src = pkgs.fetchFromGitHub {
        owner = "zsh-users";
        repo = "zsh-syntax-highlighting";
        rev = "0.7.1";
        sha256 = "03r6hpb5fy4yaakqm3lbf4xcvd408r44jgpv4lnzl9asp4sb9qc0";
      };
    }
  ];
};

# For fish shell users
programs.fish = {
  enable = true;
  
  shellAliases = {
    ll = "ls -la";
    g = "git";
  };
  
  functions = {
    fish_greeting = "echo Welcome to fish!";
  };
  
  plugins = [
    {
      name = "z";
      src = pkgs.fetchFromGitHub {
        owner = "jethrokuan";
        repo = "z";
        rev = "85f863f20f24faf675827fb00f3a4e15c7838d76";
        sha256 = "1kaa0k9d535jnzj1pnbg1jj1d37lrdvr43s33psfa5dz3ycmdqvk";
      };
    }
  ];
};
```

### Git Configuration

Set up Git with your preferences:

```nix
programs.git = {
  enable = true;
  userName = "Your Name";
  userEmail = "your.email@example.com";
  
  extraConfig = {
    init.defaultBranch = "main";
    pull.rebase = true;
    push.autoSetupRemote = true;
    
    core = {
      editor = "nvim";
      pager = "delta";
    };
    
    delta = {
      navigate = true;
      light = false;
      side-by-side = true;
    };
  };
  
  ignores = [
    ".DS_Store"
    "*.swp"
    ".idea"
    ".vscode"
    "node_modules"
  ];
  
  # Install and configure useful git aliases
  aliases = {
    co = "checkout";
    cob = "checkout -b";
    st = "status";
    br = "branch";
    unstage = "reset HEAD --";
    last = "log -1 HEAD";
    visual = "!gitk";
    cm = "commit -m";
    lg = "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit";
  };
};
```

### Neovim/Vim Configuration

Configure your text editor:

```nix
programs.neovim = {
  enable = true;
  viAlias = true;
  vimAlias = true;
  
  extraConfig = ''
    set number
    set relativenumber
    set expandtab
    set tabstop=2
    set shiftwidth=2
    set smartindent
    set cursorline
    
    " Your custom vim configuration here
  '';
  
  plugins = with pkgs.vimPlugins; [
    vim-nix
    vim-surround
    vim-commentary
    nerdtree
    vim-airline
    nord-vim
  ];
};
```

### Terminal Emulator Configuration

Configure your terminal:

```nix
programs.alacritty = {
  enable = true;
  
  settings = {
    env = {
      TERM = "xterm-256color";
    };
    
    window = {
      dimensions = {
        columns = 120;
        lines = 30;
      };
      padding = {
        x = 10;
        y = 10;
      };
      decorations = "full";
      opacity = 1.0;
    };
    
    font = {
      normal = {
        family = "JetBrainsMono Nerd Font";
        style = "Regular";
      };
      size = 14.0;
    };
    
    colors = {
      # Nord color scheme
      primary = {
        background = "#2e3440";
        foreground = "#d8dee9";
      };
    };
  };
};

# For iTerm2 users (via preference files)
home.file.".iterm2_profile.json".source = ./iterm2_profile.json;
```

### Environment Variables

Set up environment variables:

```nix
home.sessionVariables = {
  EDITOR = "nvim";
  VISUAL = "nvim";
  PAGER = "less -R";
  LANG = "en_US.UTF-8";
  LC_ALL = "en_US.UTF-8";
  PATH = "$HOME/.local/bin:$PATH";
  # Apple Silicon specific
  LDFLAGS = "-L/opt/homebrew/opt/libffi/lib";
  CPPFLAGS = "-I/opt/homebrew/opt/libffi/include";
};
```

### Dotfiles Management

Manage config files with Home Manager:

```nix
# Create .config/starship.toml
xdg.configFile."starship.toml".text = ''
  [character]
  success_symbol = "[➜](bold green)"
  error_symbol = "[✗](bold red)"
'';

# Create .tmux.conf
home.file.".tmux.conf".text = ''
  set -g default-terminal "screen-256color"
  set -g mouse on
  set -g base-index 1
  set -g status-style 'bg=#333333 fg=#5eacd3'
'';

# Link existing files
home.file.".config/myapp/config.json".source = ./configs/myapp-config.json;
```

### Application Configuration

Configure GUI applications:

```nix
# Configure VSCode
programs.vscode = {
  enable = true;
  extensions = with pkgs.vscode-extensions; [
    vscodevim.vim
    ms-python.python
    matklad.rust-analyzer
    yzhang.markdown-all-in-one
  ];
  userSettings = {
    "editor.fontSize" = 14;
    "editor.fontFamily" = "'JetBrains Mono', 'monospace'";
    "editor.formatOnSave" = true;
    "workbench.colorTheme" = "Nord";
  };
};

# Configure Karabiner Elements
home.file.".config/karabiner/karabiner.json".source = ./karabiner.json;

# Configure Rectangle (window manager)
home.file."Library/Application Support/Rectangle/RectangleConfig.json".source = ./rectangle-config.json;
```

### Apple Silicon Considerations

Specific configuration for Apple Silicon (M1/M2/M3) Macs:

```nix
# Detect architecture
let
  isAppleSilicon = pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64;

  # Special path for Homebrew on Apple Silicon
  homebrewPath = if isAppleSilicon 
    then "/opt/homebrew/bin" 
    else "/usr/local/bin";
in {
  home.sessionVariables = {
    # Add Homebrew to PATH based on architecture
    PATH = "${homebrewPath}:$PATH";
    
    # Apple Silicon specific environment variables for some packages
    LDFLAGS = if isAppleSilicon then "-L/opt/homebrew/opt/libffi/lib" else "";
    CPPFLAGS = if isAppleSilicon then "-I/opt/homebrew/opt/libffi/include" else "";
  };
  
  # ... rest of your configuration
}
```

## Advanced Usage

### Directory Structure

For larger configurations, consider this directory structure:

```
~/.config/home-manager/
├── flake.nix
├── flake.lock
├── home.nix
├── modules/
│   ├── shells.nix
│   ├── git.nix
│   ├── editors.nix
│   └── mac-apps.nix
├── configs/
│   ├── karabiner.json
│   └── rectangle-config.json
└── hosts/
    ├── work-mac.nix
    └── personal-mac.nix
```

### Multiple Profiles

Support configurations for different environments:

```nix
# In flake.nix
outputs = { nixpkgs, home-manager, ... }: {
  homeConfigurations = {
    "personal" = home-manager.lib.homeManagerConfiguration {
      pkgs = nixpkgs.legacyPackages.aarch64-darwin;
      modules = [ ./hosts/personal-mac.nix ];
    };
    
    "work" = home-manager.lib.homeManagerConfiguration {
      pkgs = nixpkgs.legacyPackages.aarch64-darwin;
      modules = [ ./hosts/work-mac.nix ];
    };
  };
};
```

Switch between profiles:

```bash
home-manager switch --flake .#personal
# or
home-manager switch --flake .#work
```

### Importing Module Files

Split your config into manageable modules:

```nix
# In home.nix
{ config, pkgs, ... }: {
  imports = [
    ./modules/shells.nix
    ./modules/git.nix
    ./modules/editors.nix
    ./modules/mac-apps.nix
  ];
  
  # Common configuration
  home.username = "yourusername";
  home.homeDirectory = "/Users/yourusername";
  
  # Enable Home Manager
  programs.home-manager.enable = true;
  
  # This value determines the Home Manager release that your
  # configuration is compatible with
  home.stateVersion = "23.11";
}
```

### Conditional Configuration

Apply different settings based on conditions:

```nix
{ config, pkgs, lib, ... }: {
  config = lib.mkMerge [
    # Base configuration
    {
      home.packages = with pkgs; [ ripgrep fd bat ];
    }
    
    # Work-specific configuration
    (lib.mkIf (builtins.getEnv "WORK_ENV" == "1") {
      home.packages = with pkgs; [ slack zoom ];
      programs.git.userEmail = "your.work.email@company.com";
    })
    
    # Personal-specific configuration
    (lib.mkIf (builtins.getEnv "WORK_ENV" != "1") {
      home.packages = with pkgs; [ spotify discord ];
      programs.git.userEmail = "your.personal.email@example.com";
    })
  ];
}
```

### Working with Secrets

Safely manage sensitive information:

```nix
{ config, pkgs, ... }: let
  # Read secret from file not tracked in git
  gitHubToken = builtins.readFile ./secrets/github-token;
in {
  programs.gh = {
    enable = true;
    settings = {
      # Use the token (note: this is still stored in the Nix store, so use caution)
      oauth_token = gitHubToken;
    };
  };
}
```

For more security, look into [sops-nix](https://github.com/Mic92/sops-nix) or [agenix](https://github.com/ryantm/agenix).

## Daily Workflow

### Applying Changes

After modifying your configuration:

**With flakes (standalone):**
```bash
home-manager switch --flake ~/.config/home-manager
```

**With nix-darwin integration:**
```bash
darwin-rebuild switch --flake ~/.config/darwin
```

### Testing Before Applying

Before applying changes:

```bash
home-manager build --flake ~/.config/home-manager
```

### News Command

Check for updates and new features:

```bash
home-manager news
```

### Generations Management

List generations:
```bash
home-manager generations
```

Roll back to a previous generation:
```bash
home-manager generations
# Find the generation you want
/nix/store/hash-home-manager-path/activate
```

## Troubleshooting

### Common Issues

1. **Missing packages in $PATH**:
   - Ensure your shell properly sources Home Manager profile
   - Check `~/.profile`, `~/.zshrc`, or `~/.bash_profile`

2. **Conflicts with Homebrew**:
   - Manage PATH order to prioritize Nix or Homebrew as desired
   - Consider which packages to manage with which system

3. **Permission errors with dotfiles**:
   - Check file ownership and permissions
   - Use `source` attribute instead of `text` for existing files

4. **Application configuration not working**:
   - Ensure correct paths to Application Support folders
   - Verify special characters are properly escaped in config files

5. **Activation script fails**:
   - Run with verbose flag: `home-manager switch -v`
   - Check for syntax errors in your Nix files

### Debugging Home Manager

For more detailed output:

```bash
HOME_MANAGER_DEBUG=1 home-manager switch --flake ~/.config/home-manager
```

Check the Home Manager logs:

```bash
less ~/.local/state/home-manager/home-manager.log
```

## Relation to Other Concepts

- **[[What is Nix]]**: Foundation for understanding the Nix ecosystem
- **[[Installing Nix on macOS]]**: Prerequisites for using Home Manager
- **[[Nix Flakes Introduction]]**: Modern approach to Nix configurations
- **[[Nix Darwin Configuration]]**: System-level configuration for macOS
- **[[Basic Nix Commands]]**: Commands used with Home Manager

## Next Steps

→ Learn about [[Working with Nix Packages]] for package management
→ Set up [[Nix Development Workflows]] for per-project environments
→ Explore [[Nix Flakes Introduction]] to fully understand flakes

---
Tags: #nix #home-manager #dotfiles #macos #user-configuration 