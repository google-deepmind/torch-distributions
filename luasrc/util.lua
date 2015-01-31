require 'torch'

distributions.util = {}

--[[
Return true if the given matrix is positive-definite, and false if it is not.
]]
function distributions.util.isposdef(m)
    local fullRank, decomposed =
        pcall(
            function()
                return torch.potrf(m):triu()
            end
        )
    return fullRank
end

--[[ Return the log-determinant of the given matrix. 
    Computed using the Cholesky decomposition if the 
    matrix is symmetric, otherwise computed using SVD ]]--
function distributions.util.logdet(m)
    local success, chol = pcall(torch.potrf, m)
    if success then
      return 2 * chol:diag():log():sum()
    else
      local u, s, v = torch.svd(m)
      return s:log():sum()
    end
end
