#include <El.hpp>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

using namespace El;

struct key_idx
{
   double key;
   int idx;
   key_idx(double k, int i) : key(k), idx(i) {}
   bool operator > (const key_idx& other) const
   {
      return (key > other.key);
   }
};

void DistHEM(DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A, double strength_tol, double stop_thresh, int max_its, std::vector<int>& match, std::vector<int>& cnode, int& num_cnode_global);

void IntraDomainMatch(int iter, DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A, std::vector< std::vector<int> >& candidates, std::vector<int>& next, std::vector<int>& match, int &n_pair, int &n_singleton, int &n_matched);

void InterDomainMatch(int iter, DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A, std::vector< std::vector<int> >& candidates, std::vector<int>& next, std::vector<int>& match, int &n_pair, int &n_singleton, int &n_matched);

void AssignCoarseNodeIndex(DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A, std::vector<int>& match, std::vector<int>& cnode, int &num_cnode_global);

void CheckMatchAndCnode(DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A, std::vector<int>& match, std::vector<int>& cnode, int num_cnode_global);

int linearSearch(int arr[], int l, int r, int x);

int
main(int argc, char* argv[])
{
    Environment env(argc, argv);
    mpi::Comm comm = mpi::NewWorldComm();
    try
    {
       const Grid grid(std::move(comm));

       DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> A(grid);
       Read(A, "S.txt");
       Matrix<double,Device::CPU>& local_mat = A.Matrix();

       if (!grid.Comm().Rank())
       {
          cout << "Global size: " << A.Height() << " x " << A.Width() << ",   ";
          cout << "Local size: " << local_mat.Height() << " x " <<  local_mat.Width() << endl;
       }

       double strength_tol = 0.8;
       double stop_thresh = 0.99;
       int max_its = 10;
       int num_cnode_global;
       std::vector<int> match, cnode;

       DistHEM(A, strength_tol, stop_thresh, max_its, match, cnode, num_cnode_global);

       CheckMatchAndCnode(A, match, cnode, num_cnode_global);
    }
    catch(std::exception& e) { ReportException(e); }

    return 0;
}

void DistHEM(DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A,
             double strength_tol,
             double stop_thresh,
             int max_its,
             std::vector<int>& match,
             std::vector<int>& cnode,
             int& num_cnode_global)
{
   Matrix<double,Device::CPU>& local_mat = A.Matrix();

   int iter = 0, flag, np, pid;
   int n_global = A.Height();
   int n_local = local_mat.Width();
   int n_pair = 0, n_singleton = 0, n_matched = 0;
   int n_pair_global = 0, n_singleton_global = 0, n_matched_global = 0;

   match.clear();
   match.resize(n_local, -1);
   cnode.clear();
   cnode.resize(n_local, -1);

   std::vector< std::vector<int> > candidates(n_local);
   std::vector<int> next(n_local, 0);

   np = A.Grid().Comm().Size();
   pid = A.Grid().Comm().Rank();

   std::vector<key_idx> pairs;

   // for each node, find its matching candidate list
   for (int i = 0; i < n_local; i++)
   {
      pairs.clear();

      for (int j = 0; j < n_global; j++)
      {
         double aji = local_mat.Get(j, i);
         if (A.GlobalCol(i) != j && aji > strength_tol)
         {
            key_idx x(aji, j);
            pairs.push_back(x);
         }
      }

      sort(pairs.begin(), pairs.end(), std::greater<key_idx>());

      for (std::vector<key_idx>::iterator it = pairs.begin(); it != pairs.end(); ++it)
      {
         candidates[i].push_back(it->idx);
      }
   }

   while (1)
   {
      if (iter)
      {
         mpi::AllReduce(&n_matched, &n_matched_global, 1, mpi::SUM, A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

         // DON'T NEED THESE TWO ALL_REDUCE. JUST FOR PRINTING STATS
         mpi::AllReduce(&n_pair, &n_pair_global, 1, mpi::SUM, A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

         mpi::AllReduce(&n_singleton, &n_singleton_global, 1, mpi::SUM, A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));
      }

      if (!pid && iter)
      {
         printf(" Iter %4d:   n_global %5d,   matched %5d,   pair %5d,   singleton %5d \n", iter,
                n_global, n_matched_global, n_pair_global/2, n_singleton_global);
      }

      iter++;

      if (iter > max_its)
      {
         flag = 1;
         break;
      }

      if (n_matched_global >= stop_thresh * n_global)
      {
         flag = 2;
         break;
      }

      IntraDomainMatch(iter, A, candidates, next, match, n_pair, n_singleton, n_matched);

      InterDomainMatch(iter, A, candidates, next, match, n_pair, n_singleton, n_matched);
   }

   if (flag == 1)
   {
      if (!pid)
      {
         printf(" Reached max # iterations %d, %.2f nodes have been matched\n",
                max_its, (n_matched_global+0.0)/n_global);
      }
   }
   else if (flag == 2)
   {
      if (!pid)
      {
         printf(" %d nodes out of %d (%.2f) have been matched in %d iterations\n",
                n_matched_global, n_global, (n_matched_global+0.0)/n_global, iter-1);
      }
   }

   /* put left overs as singletons */
   for (int i = 0; i < n_local; i++)
   {
      if (match[i] == -1)
      {
         match[i] = A.GlobalCol(i);
         n_singleton ++;
      }
   }
   n_matched = n_pair + n_singleton;

   assert(n_matched == n_local);

   AssignCoarseNodeIndex(A, match, cnode, num_cnode_global);

   if (!pid)
   {
      printf(" global number of C-pts %d, coarsening factor %.2f\n",
            num_cnode_global, (n_global + 0.0) / num_cnode_global);
   }
}

/* local on-process greedy (sequential) matching */
/* in this version, we first try to find for each node the next unmatched
 * one locally, then find the next external node to send a request */
void IntraDomainMatch(int iter,
                      DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A,
                      std::vector< std::vector<int> >& candidates,
                      std::vector<int>& next,
                      std::vector<int>& match,
                      int &n_pair,
                      int &n_singleton,
                      int &n_matched)
{
   Matrix<double,Device::CPU>& local_mat = A.Matrix();

   int i,j, np, pid;
   int n_local = local_mat.Width();

   np = A.Grid().Comm().Size();
   pid = A.Grid().Comm().Rank();

   for (i = 0; i < n_local; i++)
   {
      if (match[i] != -1)
      {
         continue;
      }

      for (j = next[i]; j < candidates[i].size(); j++)
      {
         int cand_j = candidates[i][j];

         if (cand_j == -1)
         {
            continue;
         }

         assert(cand_j != A.GlobalCol(i));

         if (A.ColOwner(cand_j) == pid)
         {
            int cand_j_local = A.LocalCol(cand_j);

            if (match[cand_j_local] == -1)
            {
               match[i] = cand_j;
               match[cand_j_local] = A.GlobalCol(i);
               n_pair += 2;
               break;
            }
         }
         else
         {
            break;
         }
      }

      next[i] = j;
   }

   for (i = 0; i < n_local; i++)
   {
      if (match[i] != -1)
      {
         continue;
      }

      if (next[i] == candidates[i].size())
      {
         match[i] = A.GlobalCol(i);
         n_singleton ++;
      }
   }

   n_matched = n_pair + n_singleton;
}

void InterDomainMatch(int iter,
                      DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A,
                      std::vector< std::vector<int> >& candidates,
                      std::vector<int>& next,
                      std::vector<int>& match,
                      int &n_pair,
                      int &n_singleton,
                      int &n_matched)
{
   Matrix<double,Device::CPU>& local_mat = A.Matrix();

   int i, j, np, pid;

   int n_global = A.Height();
   int n_local = local_mat.Width();

   np = A.Grid().Comm().Size();
   pid = A.Grid().Comm().Rank();

   std::vector<int> requests_send_counts(np, 0);
   std::vector<int> requests_send_starts(np+1);
   std::vector<int> requests_recv_counts(np);
   std::vector<int> requests_recv_starts(np+1);

   /* fill requests */
   for (i = 0; i < n_local; i++)
   {
      if (match[i] != -1)
      {
         continue;
      }

      j = next[i];

      assert(j >= 0 && j < candidates[i].size());

      int cand_j = candidates[i][j];

      assert(cand_j != -1);

      int p = A.ColOwner(cand_j);

      assert(p != pid);

      /* each request contains 2 integer numbers */
      requests_send_counts[p] += 2;
   }

   requests_send_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      requests_send_starts[i+1] = requests_send_starts[i] + requests_send_counts[i];
   }

   std::vector<int> requests_send(requests_send_starts[np]);

   for (i = 0; i < n_local; i++)
   {
      if (match[i] != -1)
      {
         continue;
      }

      j = next[i];
      int cand_j = candidates[i][j];
      int p = A.ColOwner(cand_j);
      int k = requests_send_starts[p];
      requests_send[k] = cand_j;
      requests_send[k+1] = A.GlobalCol(i);
      requests_send_starts[p] = k + 2;
   }

   assert(requests_send_starts[np] == requests_send_starts[np-1]);

   for (i = np; i > 0; i--)
   {
      requests_send_starts[i] = requests_send_starts[i-1];
   }
   requests_send_starts[0] = 0;

   /* send/recv requests */
   mpi::AllToAll(requests_send_counts.data(), 1, requests_recv_counts.data(), 1,
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   requests_recv_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      requests_recv_starts[i+1] = requests_recv_starts[i] + requests_recv_counts[i];
   }

   std::vector<int> requests_recv(requests_recv_starts[np]);

   mpi::AllToAll(requests_send.data(), requests_send_counts.data(), requests_send_starts.data(),
                 requests_recv.data(), requests_recv_counts.data(), requests_recv_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   std::vector<int> responses_send(requests_recv_starts[np]);
   std::vector<int> responses_recv(requests_send_starts[np]);

   /* store the strongest weight and node idx of which makes request to a node */
   std::vector<double> max_req_weight(n_local, -1.0);
   std::vector<int> max_req_gindex(n_local, -1);

   /* fill responses */
   for (i = 0; i < requests_recv_starts[np]; i+=2)
   {
      int req_to = A.LocalCol(requests_recv[i]);

      assert(req_to >= 0 && req_to < n_local);

      if (match[req_to] != -1)
      {
         continue;
      }

      int req_from = requests_recv[i+1];

      if (local_mat.Get(req_from, req_to) > max_req_weight[req_to])
      {
         max_req_weight[req_to] = local_mat.Get(req_from, req_to);
         max_req_gindex[req_to] = req_from;
      }
   }

   for (i = 0; i < requests_recv_starts[np]; i+=2)
   {
      int req_to = A.LocalCol(requests_recv[i]);
      int req_from = requests_recv[i+1];

      if (match[req_to] == -1 && max_req_gindex[req_to] == req_from)
      {
         responses_send[i] = 1;
      }
      else
      {
         responses_send[i] = 0;
      }
      /* reserved place */
      responses_send[i+1] = -1;
   }

   /* send/recv responses */
   mpi::AllToAll(responses_send.data(), requests_recv_counts.data(), requests_recv_starts.data(),
                 responses_recv.data(), requests_send_counts.data(), requests_send_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   std::vector<int> resp(n_local, -1);

   for (i = 0; i < requests_send_starts[np]; i+=2)
   {
      int req_from = A.LocalCol(requests_send[i+1]);

      assert(req_from >= 0 && req_from < n_local);

      resp[req_from] = responses_recv[i];
   }

   requests_send_counts.clear();
   requests_send_starts.clear();
   requests_recv_counts.clear();
   requests_recv_starts.clear();
   requests_send.clear();
   requests_recv.clear();
   responses_send.clear();
   responses_recv.clear();

   /* find tentative matches based on requests and responses */
   int num_ext_index = 0;
   std::vector<int> ext_index;

   for (i = 0; i < n_local; i++)
   {
      if (match[i] != -1)
      {
         continue;
      }

      j = next[i];
      int cand_j = candidates[i][j];

      assert(resp[i] == 0 || resp[i] == 1);

      /* use value <= -2 to mark this is a tentative match */
      int tentative_match = 0;

      if (resp[i] == 1 && max_req_gindex[i] != -1)
      {
         /* break tie arbitrarily */
         if ( (A.GlobalCol(i) + iter) % 2 )
         {
            tentative_match = -2 - cand_j;
         }
         else
         {
            tentative_match = -2 - max_req_gindex[i];
         }
      }
      else if (resp[i] == 1)
      {
         tentative_match = -2 - cand_j;
      }
      else if (max_req_gindex[i] != -1)
      {
         tentative_match = -2 - max_req_gindex[i];
      }

      if (tentative_match)
      {
         match[i] = tentative_match;
         j = -tentative_match - 2;

         assert(A.ColOwner(j) != pid);

         ext_index.push_back(j);
      }
   }

   num_ext_index = ext_index.size();

   max_req_weight.clear();
   max_req_gindex.clear();
   resp.clear();

   /* to retrieve off-proc match information */
   sort(ext_index.begin(), ext_index.end());

   j = 0;
   for (i = 0; i < num_ext_index; i++)
   {
      if (i == 0 || ext_index[i] != ext_index[i-1])
      {
         ext_index[j++] = ext_index[i];
      }
   }
   num_ext_index = j;

   std::vector<int> index_send_counts(np, 0);
   std::vector<int> index_send_starts(np+1);

   std::vector<int> index_recv_counts(np);
   std::vector<int> index_recv_starts(np+1);

   for (i = 0; i < num_ext_index; i++)
   {
      j = ext_index[i];

      int p = A.ColOwner(j);

      assert(p != pid);

      index_send_counts[p] ++;
   }

   index_send_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      index_send_starts[i+1] = index_send_starts[i] + index_send_counts[i];
   }

   assert(index_send_starts[np] == num_ext_index);

   std::vector<int> index_send(index_send_starts[np]);

   for (i = 0; i < num_ext_index; i++)
   {
      j = ext_index[i];
      int p = A.ColOwner(j);
      index_send[index_send_starts[p]++] = j;
   }

   assert(index_send_starts[np] == index_send_starts[np-1]);

   for (i = np; i > 0; i--)
   {
      index_send_starts[i] = index_send_starts[i-1];
   }
   index_send_starts[0] = 0;

   mpi::AllToAll(index_send_counts.data(), 1, index_recv_counts.data(), 1,
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   index_recv_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      index_recv_starts[i+1] = index_recv_starts[i] + index_recv_counts[i];
   }

   std::vector<int> index_recv(index_recv_starts[np]);

   mpi::AllToAll(index_send.data(), index_send_counts.data(), index_send_starts.data(),
                 index_recv.data(), index_recv_counts.data(), index_recv_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   std::vector<int> match_send(index_recv_starts[np]);
   std::vector<int> match_recv(index_send_starts[np]);

   for (i = 0; i < index_recv_starts[np]; i++)
   {
      j = A.LocalCol(index_recv[i]);

      assert(j >= 0 && j < n_local);

      match_send[i] = match[j];
   }

   mpi::AllToAll(match_send.data(), index_recv_counts.data(), index_recv_starts.data(),
                 match_recv.data(), index_send_counts.data(), index_send_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   /* find successful matches from the tentative matches */
   for (i = 0; i < n_local; i++)
   {
      if (match[i] > -2)
      {
         continue;
      }

      j = -match[i] - 2;

      int p = linearSearch(index_send.data(), 0, index_send_starts[np]-1, j);

      assert(p != -1);

      int k = match_recv[p];

      assert(k != -1);

      if (k <= -2 && -k-2 == A.GlobalCol(i))
      {
         match[i] = j;

         assert(A.GlobalCol(i) != j);

         n_pair ++;
      }
      else
      {
         match[i] = -1;
      }
   }

   n_matched = n_pair + n_singleton;

   index_send.clear();
   index_recv.clear();
   match_send.clear();
   match_recv.clear();
   ext_index.clear();

   /* retrieve off-proc match information again to update candidates list */
   num_ext_index = 0;
   for (i = 0; i < n_local; i++)
   {
      assert(match[i] >= -1 && match[i] < n_global);

      if (match[i] == -1)
      {
         for (j = next[i]; j < candidates[i].size(); j++)
         {
            int cand_j = candidates[i][j];

            if (cand_j == -1)
            {
               continue;
            }

            if (A.ColOwner(cand_j) != pid)
            {
               ext_index.push_back(cand_j);
            }
         }
      }
   }

   num_ext_index = ext_index.size();

   sort(ext_index.begin(), ext_index.end());

   j = 0;
   for (i = 0; i < num_ext_index; i++)
   {
      if (i == 0 || ext_index[i] != ext_index[i-1])
      {
         ext_index[j++] = ext_index[i];
      }
   }
   num_ext_index = j;

   for (i = 0; i < np; i++)
   {
      index_send_counts[i] = 0;
   }

   for (i = 0; i < num_ext_index; i++)
   {
      j = ext_index[i];

      int p = A.ColOwner(j);

      assert(p != pid);

      index_send_counts[p] ++;
   }

   index_send_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      index_send_starts[i+1] = index_send_starts[i] + index_send_counts[i];
   }

   assert(index_send_starts[np] == num_ext_index);

   index_send.resize(index_send_starts[np]);

   for (i = 0; i < num_ext_index; i++)
   {
      j = ext_index[i];
      int p = A.ColOwner(j);
      index_send[index_send_starts[p]++] = j;
   }

   assert(index_send_starts[np] == index_send_starts[np-1]);

   for (i = np; i > 0; i--)
   {
      index_send_starts[i] = index_send_starts[i-1];
   }
   index_send_starts[0] = 0;

   mpi::AllToAll(index_send_counts.data(), 1, index_recv_counts.data(), 1,
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   index_recv_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      index_recv_starts[i+1] = index_recv_starts[i] + index_recv_counts[i];
   }

   index_recv.resize(index_recv_starts[np]);

   mpi::AllToAll(index_send.data(), index_send_counts.data(), index_send_starts.data(),
                 index_recv.data(), index_recv_counts.data(), index_recv_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   match_send.resize(index_recv_starts[np]);
   match_recv.resize(index_send_starts[np]);

   for (i = 0; i < index_recv_starts[np]; i++)
   {
      j = A.LocalCol(index_recv[i]);

      assert(j >= 0 && j < n_local);

      match_send[i] = match[j];
   }

   mpi::AllToAll(match_send.data(), index_recv_counts.data(), index_recv_starts.data(),
                 match_recv.data(), index_send_counts.data(), index_send_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   for (i = 0; i < n_local; i++)
   {
      if (match[i] == -1)
      {
         for (j = next[i]; j < candidates[i].size(); j++)
         {
            int cand_j = candidates[i][j];

            if (cand_j == -1)
            {
               continue;
            }

            if (A.ColOwner(cand_j) == pid)
            {
               if (match[A.LocalCol(cand_j)] != -1)
               {
                  candidates[i][j] = -1;
               }
            }
            else
            {
               int p = linearSearch(index_send.data(), 0, index_send_starts[np]-1, cand_j);

               assert(p != -1);

               int k = match_recv[p];

               assert(k >= -1);

               if (k != -1)
               {
                  candidates[i][j] = -1;
               }
            }
         }
      }
   }

   /*
   printf(" Iter %4d, Pid %d: (Intra) matched %5d, pair %5d, single %5d; (Inter) matched %5d, pair %5d, single %5d\n",
          iter, pid, n_matched_in, n_pair_in, n_singleton_in, *n_matched, *n_pair, *n_singleton);
   */
}


int linearSearch(int arr[], int l, int r, int x)
{
   int i, p = -1, flag = 0;
   for (i = l; i <= r; i++)
   {
      if (arr[i] == x)
      {
         flag++;
         p = i;
      }
   }

   if (flag > 1)
   {
      printf(" linearSearch: found more than one\n");
   }

   return p;
}

void AssignCoarseNodeIndex(DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A,
                           std::vector<int>& match,
                           std::vector<int>& cnode,
                           int &num_cnode_global)
{
   Matrix<double,Device::CPU>& local_mat = A.Matrix();

   int i, j, np, pid, cnum = 0, cnum_presum = 0;
   int n_local = local_mat.Width();

   np = A.Grid().Comm().Size();
   pid = A.Grid().Comm().Rank();

   std::vector<int> index_send_counts(np, 0);
   std::vector<int> index_send_starts(np+1);
   std::vector<int> index_recv_counts(np);
   std::vector<int> index_recv_starts(np+1);

   for (i = 0; i < n_local; i++)
   {
      j = match[i];
      int p = A.ColOwner(j);

      if ( p > pid || (p == pid && i <= A.LocalCol(j)) )
      {
         cnode[i] = cnum++;
      }

      //if (j < A.GlobalCol(i) && p != pid)
      if (p < pid)
      {
         index_send_counts[p] ++;
      }
   }

   mpi::Scan(&cnum, &cnum_presum, 1, mpi::SUM, A.Grid().Comm(),
             SyncInfoFromMatrix(A.LockedMatrix()));

   cnum_presum -= cnum;

   for (i = 0; i < n_local; i++)
   {
      if (cnode[i] != -1)
      {
         cnode[i] += cnum_presum;
      }
   }

   mpi::AllReduce(&cnum, &num_cnode_global, 1, mpi::SUM,
                  A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   index_send_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      index_send_starts[i+1] = index_send_starts[i] + index_send_counts[i];
   }

   std::vector<int> index_send(index_send_starts[np]);

   for (i = 0; i < n_local; i++)
   {
      j = match[i];
      int p = A.ColOwner(j);
      //if (j < A.GlobalCol(i) && p != pid)
      if (p < pid)
      {
         index_send[index_send_starts[p]++] = j;
      }
   }

   assert(index_send_starts[np] == index_send_starts[np-1]);

   for (i = np; i > 0; i--)
   {
      index_send_starts[i] = index_send_starts[i-1];
   }
   index_send_starts[0] = 0;

   mpi::AllToAll(index_send_counts.data(), 1, index_recv_counts.data(), 1,
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));


   index_recv_starts[0] = 0;
   for (i = 0; i < np; i++)
   {
      index_recv_starts[i+1] = index_recv_starts[i] + index_recv_counts[i];
   }

   std::vector<int> index_recv(index_recv_starts[np]);

   mpi::AllToAll(index_send.data(), index_send_counts.data(), index_send_starts.data(),
                 index_recv.data(), index_recv_counts.data(), index_recv_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   std::vector<int> cnode_send(index_recv_starts[np]);
   std::vector<int> cnode_recv(index_send_starts[np]);

   for (i = 0; i < index_recv_starts[np]; i++)
   {
      j = A.LocalCol(index_recv[i]);

      assert(j >= 0 && j < n_local);

      assert(cnode[j] != -1);

      cnode_send[i] = cnode[j];
   }

   mpi::AllToAll(cnode_send.data(), index_recv_counts.data(), index_recv_starts.data(),
                 cnode_recv.data(), index_send_counts.data(), index_send_starts.data(),
                 A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   for (i = 0; i < n_local; i++)
   {
      if (cnode[i] == -1)
      {
         j = match[i];
         if (A.ColOwner(j) == pid)
         {
            cnode[i] = cnode[A.LocalCol(j)];
         }
         else
         {
            int p = linearSearch(index_send.data(), 0, index_send_starts[np]-1, j);

            assert(p != -1);

            cnode[i] = cnode_recv[p];
         }
      }
   }
}

void CheckMatchAndCnode(DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>& A,
                        std::vector<int>& match,
                        std::vector<int>& cnode,
                        int num_cnode_global)
{
   int np, pid;
   std::vector<int> recvcounts, displs, match_global, cnode_global, cnode_seq;
   int cnum = 0;
   int n_pair = 0, n_singleton = 0;
   double matched_edge_weights = 0.0;

   Matrix<double,Device::CPU>& local_mat = A.Matrix();

   int n_global = A.Height();
   int n_local = local_mat.Width();

   np = A.Grid().Comm().Size();
   pid = A.Grid().Comm().Rank();

   if (!pid)
   {
      recvcounts.resize(np);
      displs.resize(np);
      match_global.resize(n_global, -1);
      cnode_global.resize(n_global, -1);
      cnode_seq.resize(n_global, -1);
   }

   mpi::Gather(&n_local, 1, recvcounts.data(), 1, 0, A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   if (!pid)
   {
      displs[0] = 0;
      for (int i = 1; i < np; i++)
      {
         displs[i] = displs[i-1] + recvcounts[i-1];
      }
   }

   // NOTE the order here: e.g np = 4, [0, 4, 8, ...., 1, 5, 9, ..., 2, 6, 10, ...., 3, 7, 11, ...]
   mpi::Gather(match.data(), n_local, match_global.data(), recvcounts.data(), displs.data(),
               0, A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));
   mpi::Gather(cnode.data(), n_local, cnode_global.data(), recvcounts.data(), displs.data(),
               0, A.Grid().Comm(), SyncInfoFromMatrix(A.LockedMatrix()));

   // TODO
   DistMatrix<double,STAR,STAR,ELEMENT,Device::CPU> A_STAR_STAR(A);

   if (!pid)
   {
      for (int i = 0; i < n_global; i++)
      {
         int j, p, pi, pj;

         p = A.ColOwner(i); pi = displs[p] + A.LocalCol(i, p);
         j = match_global[pi];
         p = A.ColOwner(j); pj = displs[p] + A.LocalCol(j, p);

         assert(j >= 0 && j < n_global);
         assert(match_global[pi] == j);
         assert(match_global[pj] == i);
         assert(cnode_global[pi] == cnode_global[pj]);

         if (i == j)
         {
            n_singleton++;
         }
         else if (i < j)
         {
            n_pair++;
            matched_edge_weights += A_STAR_STAR.GetLocal(i,j);
         }
      }

      for (int i = 0; i < n_global; i++)
      {
         int j, p, pj;

         if (cnode_seq[i] != -1)
         {
            continue;
         }
         j = match_global[i];
         p = A.ColOwner(j); pj = displs[p] + A.LocalCol(j, p);
         cnode_seq[i] = cnode_seq[pj] = cnum++;
      }

      for (int i = 0; i < n_global; i++)
      {
         assert(cnode_seq[i] == cnode_global[i]);
      }

      assert(cnum == num_cnode_global);
      printf(" Num of pairs %d, num of singletons %d, Avg matched edge weight %f\n", n_pair, n_singleton, matched_edge_weights / n_pair);
   }
}

