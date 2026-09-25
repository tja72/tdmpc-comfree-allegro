-- Drop HTML comments (claim IDs, PENDING/FALLBACK markers) from the rendered page.
-- They stay in sections/*.qmd, where the editing scripts grep for them.
local function is_comment(el)
  return el.format:match("html") and el.text:match("^%s*<!%-%-.*%-%->%s*$")
end

function RawInline(el)
  if is_comment(el) then return {} end
end

function RawBlock(el)
  if is_comment(el) then return {} end
end
