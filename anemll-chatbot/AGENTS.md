# Repository Guidelines

## Project Structure & Modules
- App code lives in `anemll-chatbot/` with `Services/`, `Views/`, `Models/`, `Extensions/`, `Resources/`, and `Assets.xcassets`. Entry point: `anemll_chatbotApp.swift`.
- Tests: `anemll-chatbotTests/` (unit, Swift Testing) and `anemll-chatbotUITests/` (UI, XCTest).
- Xcode project: `anemll-chatbot.xcodeproj` with shared schemes: `anemll-chatbot` (iOS) and `anemll-chatbox macOS`.
- Runtime artifacts (not in repo): `models.json` and downloaded models under the app’s Documents directory (`Documents/Models`).

## Build, Test, and Run
- Open in Xcode: `xed .` or `open anemll-chatbot.xcodeproj`.
- List schemes: `xcodebuild -list -project anemll-chatbot.xcodeproj`.
- iOS build: `xcodebuild -scheme "anemll-chatbot" -destination 'platform=iOS Simulator,name=iPhone 15' build`.
- iOS tests: `xcodebuild -scheme "anemll-chatbot" -destination 'platform=iOS Simulator,name=iPhone 15' test`.
- macOS build: `xcodebuild -scheme "anemll-chatbox macOS" -destination 'platform=macOS' build`.
- After cloning, let Xcode resolve SwiftPM packages for `AnemllCore`, `Yams`, etc.

## Coding Style & Naming
- Swift 5.9+; 4‑space indentation; soft 120‑column limit; avoid trailing whitespace.
- Types `PascalCase`; properties/functions `lowerCamelCase`; enum cases `lowerCamelCase`.
- One primary type per file named `TypeName.swift`. Views end with `View` (e.g., `ChatView.swift`); services end with `Service` (e.g., `ModelService.swift`).
- No enforced linter; use Xcode’s formatter (Editor → Structure → Re‑Indent). Prefer explicit access control and doc comments for public APIs.

## Testing Guidelines
- Unit tests use Swift Testing (`import Testing`, `@Test`) in `anemll-chatbotTests`.
- UI tests use XCTest in `anemll-chatbotUITests`.
- Name files `FeatureNameTests.swift`. Example: run a subset: `xcodebuild test -scheme "anemll-chatbot" -only-testing:anemll-chatbotTests/FeatureNameTests -destination 'platform=iOS Simulator,name=iPhone 15'`.
- Add tests for new logic and edge cases; prefer small, deterministic tests.

## Commit & PR Guidelines
- Commits: short imperative subject, optional body with rationale (e.g., “Refactor InferenceService token handling”).
- PRs: clear description, linked issues, steps to verify, and UI screenshots/GIFs when applicable. Note any changes to model storage or `models.json` behavior.

## Security & Configuration Tips
- Do not commit model binaries or user chat logs. Keep repo changes limited to source and project files.
- Network access to Hugging Face is optional; the app should run without credentials.
- Keep `.xcodeproj` edits minimal; add files via Xcode to avoid project diffs.
