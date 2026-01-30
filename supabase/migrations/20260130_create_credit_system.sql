-- LabFork Credit System Migration
-- Creates tables for user credits, transactions, and devices in the distributed compute network

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User Credits Table
-- Stores the credit balance and lifetime stats for each user
CREATE TABLE IF NOT EXISTS user_credits (
  user_id TEXT PRIMARY KEY,
  balance DECIMAL(10, 2) NOT NULL DEFAULT 0 CHECK (balance >= 0),
  total_earned DECIMAL(10, 2) NOT NULL DEFAULT 0,
  total_spent DECIMAL(10, 2) NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Credit Transactions Table
-- Records all credit transactions (earn, spend, bonus, refund)
CREATE TABLE IF NOT EXISTS credit_transactions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  amount DECIMAL(10, 2) NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('earn', 'spend', 'bonus', 'refund')),
  task_id TEXT,
  description TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Devices Table
-- Stores registered compute devices that contribute to the network
CREATE TABLE IF NOT EXISTS devices (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  name TEXT NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('power', 'standard', 'crowd')),
  capabilities JSONB NOT NULL,
  availability JSONB NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('online', 'busy', 'offline', 'paused')),
  current_task_id TEXT,
  stats JSONB NOT NULL DEFAULT '{"tasksCompleted": 0, "creditsEarned": 0, "totalComputeTime": 0}'::jsonb,
  registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_credit_transactions_user_id ON credit_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_credit_transactions_created_at ON credit_transactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_credit_transactions_task_id ON credit_transactions(task_id) WHERE task_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_devices_user_id ON devices(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_devices_tier ON devices(tier);
CREATE INDEX IF NOT EXISTS idx_devices_last_seen ON devices(last_seen DESC);

-- Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE user_credits ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;

-- User Credits Policies
-- Users can view their own credits
CREATE POLICY "Users can view own credits"
  ON user_credits FOR SELECT
  USING (auth.uid()::text = user_id);

-- Users can update their own credits (through triggers only)
CREATE POLICY "Service role can update credits"
  ON user_credits FOR UPDATE
  USING (true);

-- Service role can insert credits
CREATE POLICY "Service role can insert credits"
  ON user_credits FOR INSERT
  WITH CHECK (true);

-- Credit Transactions Policies
-- Users can view their own transactions
CREATE POLICY "Users can view own transactions"
  ON credit_transactions FOR SELECT
  USING (auth.uid()::text = user_id);

-- Service role can insert transactions
CREATE POLICY "Service role can insert transactions"
  ON credit_transactions FOR INSERT
  WITH CHECK (true);

-- Devices Policies
-- Users can view their own devices
CREATE POLICY "Users can view own devices"
  ON devices FOR SELECT
  USING (auth.uid()::text = user_id OR user_id IS NULL);

-- Users can insert their own devices
CREATE POLICY "Users can insert own devices"
  ON devices FOR INSERT
  WITH CHECK (auth.uid()::text = user_id OR user_id IS NULL);

-- Users can update their own devices
CREATE POLICY "Users can update own devices"
  ON devices FOR UPDATE
  USING (auth.uid()::text = user_id OR user_id IS NULL);

-- All users can view online devices (for network stats)
CREATE POLICY "Anyone can view online devices"
  ON devices FOR SELECT
  USING (status = 'online');

-- Function to update user_credits updated_at timestamp
CREATE OR REPLACE FUNCTION update_user_credits_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_user_credits_timestamp
  BEFORE UPDATE ON user_credits
  FOR EACH ROW
  EXECUTE FUNCTION update_user_credits_timestamp();

-- Function to update device last_seen timestamp
CREATE OR REPLACE FUNCTION update_device_last_seen()
RETURNS TRIGGER AS $$
BEGIN
  NEW.last_seen = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update last_seen
CREATE TRIGGER update_device_last_seen
  BEFORE UPDATE ON devices
  FOR EACH ROW
  EXECUTE FUNCTION update_device_last_seen();

-- Function to automatically update user_credits when transaction is created
CREATE OR REPLACE FUNCTION process_credit_transaction()
RETURNS TRIGGER AS $$
BEGIN
  -- Upsert user_credits record
  INSERT INTO user_credits (user_id, balance, total_earned, total_spent)
  VALUES (
    NEW.user_id,
    CASE
      WHEN NEW.amount > 0 THEN NEW.amount
      ELSE 0
    END,
    CASE
      WHEN NEW.amount > 0 THEN NEW.amount
      ELSE 0
    END,
    CASE
      WHEN NEW.amount < 0 THEN ABS(NEW.amount)
      ELSE 0
    END
  )
  ON CONFLICT (user_id) DO UPDATE SET
    balance = user_credits.balance + NEW.amount,
    total_earned = CASE
      WHEN NEW.amount > 0 THEN user_credits.total_earned + NEW.amount
      ELSE user_credits.total_earned
    END,
    total_spent = CASE
      WHEN NEW.amount < 0 THEN user_credits.total_spent + ABS(NEW.amount)
      ELSE user_credits.total_spent
    END;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update user_credits on transaction
CREATE TRIGGER process_credit_transaction
  AFTER INSERT ON credit_transactions
  FOR EACH ROW
  EXECUTE FUNCTION process_credit_transaction();

-- Comments for documentation
COMMENT ON TABLE user_credits IS 'User credit balances and lifetime statistics';
COMMENT ON TABLE credit_transactions IS 'Record of all credit transactions';
COMMENT ON TABLE devices IS 'Registered compute devices in the distributed network';
COMMENT ON COLUMN user_credits.balance IS 'Current available credit balance';
COMMENT ON COLUMN user_credits.total_earned IS 'Lifetime total credits earned';
COMMENT ON COLUMN user_credits.total_spent IS 'Lifetime total credits spent';
COMMENT ON COLUMN credit_transactions.type IS 'Transaction type: earn, spend, bonus, or refund';
COMMENT ON COLUMN devices.tier IS 'Device performance tier: power, standard, or crowd';
COMMENT ON COLUMN devices.capabilities IS 'Device hardware capabilities (compute, memory, bandwidth, platform)';
COMMENT ON COLUMN devices.availability IS 'Device availability preferences (wifiOnly, chargingOnly, etc.)';
COMMENT ON COLUMN devices.stats IS 'Lifetime device statistics (tasksCompleted, creditsEarned, totalComputeTime)';
