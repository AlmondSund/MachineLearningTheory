# Control

Control code owns the adapter from policy action state to actual game or
operating-system input.

The module consumes a binary held-state action:

- `0`: release input
- `1`: hold input

It should apply the requested state predictably and expose enough error
information for the runtime loop to log or fail safely. Do not hide failed input
application, missed deadlines, or permission errors.

Control is a side-effect boundary. Model code should not call OS input APIs
directly.
