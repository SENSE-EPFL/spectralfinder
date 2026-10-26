// Typing in a table dropdown's search box (e.g. a gas name): Dash's table catches Backspace and
// Delete to clear the whole cell, and the arrow / Home / End keys to move to another cell, so a
// typo couldn't be corrected -- the box stayed stuck on the wrong text. Those keys now stay with
// the search box: stopped here, at the very start of the event's path (window, capture phase),
// before the table sees them. Their default action (deleting a character, moving the cursor) is
// left alone. Up / Down / Enter / Escape still reach the dropdown, to pick an option.
window.addEventListener("keydown", (event) => {
    const editingKeys = ["Backspace", "Delete", "ArrowLeft", "ArrowRight", "Home", "End"];
    const box = event.target;
    if (editingKeys.includes(event.key) && box instanceof HTMLInputElement
            && box.closest(".dash-spreadsheet .Select-input")) {
        event.stopPropagation();
    }
}, true);
